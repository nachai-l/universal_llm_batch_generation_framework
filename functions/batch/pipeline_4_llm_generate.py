# functions/batch/pipeline_4_llm_generate.py
"""
Pipeline 4 — LLM Batch Generation (+ optional judge, auto-retry on judge fail)

Intent
------
Execute LLM generation deterministically for each WorkItem produced by Pipeline 3,
optionally run a judge prompt, and write one output file per accepted LLM call.

Supports parallel execution via max_workers parameter.

Outputs
-------
- artifacts/cache/llm_outputs/{cache_id}.json
- artifacts/cache/llm_failures/{cache_id}__a{attempt}.json
- artifacts/cache/pipeline4_manifest.json

Notes
-----
- Pipeline 3 may provide deduplicated group contexts:
    - pipeline3_group_contexts.json : group_context_id -> context string
    - WorkItems may reference group_context_id (do NOT repeat full context in each item)
- WorkItems may also contain inline context (row-wise / group_output modes).
- We intentionally do NOT treat runner-level caching as the canonical cache because judge-failed
  generations must not be cached as "final". We treat llm_outputs/*.json as the canonical cache.
- Parallel execution (max_workers > 1) uses ThreadPoolExecutor for I/O-bound LLM calls.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from functions.core.llm_artifacts_cache import WorkRef, load_group_context_map, parse_pipeline3_items
from functions.core.llm_batch_engine import generate_with_optional_judge
from functions.core.llm_batch_storage import pre_scan_cache, write_failure_json
from functions.core.schema_runtime import resolve_schema_models
from functions.io.readers import read_json
from functions.io.writers import write_json
from functions.llm.runner import run_prompt_yaml_json
from functions.utils.config import load_parameters
from functions.utils.hashing import sha1_file
from functions.utils.logging import configure_logging_from_params, get_logger
from functions.utils.paths import repo_root_from_parameters_path, resolve_path
from functions.utils.verbosity import VerbosityLogger, clamp_verbose, item_log_every_n


PIPELINE3_WORK_ITEMS_PATH_DEFAULT = "artifacts/cache/pipeline3_work_items.json"
PIPELINE3_GROUP_CONTEXTS_PATH_DEFAULT = "artifacts/cache/pipeline3_group_contexts.json"

PIPELINE4_MANIFEST_PATH_DEFAULT = "artifacts/cache/pipeline4_manifest.json"

LLM_OUTPUTS_DIR_DEFAULT = "artifacts/cache/llm_outputs"
LLM_FAILURES_DIR_DEFAULT = "artifacts/cache/llm_failures"


@dataclass(frozen=True)
class WorkItemResult:
    """Result of processing a single WorkItem."""
    item_no: int
    work_id: str
    cache_id: str
    status: str  # "cache_hit" | "success" | "failure"
    output_file: Optional[str]  # filename in outputs or failures dir
    elapsed_sec: float
    error_msg: Optional[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _silence_writers_logger() -> None:
    """Silence functions.io.writers INFO logs for this run (keep WARNING+)."""
    lg = logging.getLogger("functions.io.writers")
    lg.setLevel(logging.WARNING)
    lg.propagate = True


# -----------------------------
# Client creation (thread-safe)
# -----------------------------

_CLIENT_TLS = threading.local()


def _build_client_bundle(params: Any, *, credentials_cfg: Any, model_name: str) -> Dict[str, Any]:
    """
    Backward-compatible client builder.

    Supports TWO shapes (so tests can monkeypatch either):
    A) Legacy: functions.llm.client_factory.build_gemini_client(silence_logs=...) -> client_obj
    B) Current: functions.llm.client.build_gemini_client(credentials_cfg, model_name_override=...) -> {"client":..., "model_name":...}

    Returns:
      {"client": <obj>, "model_name": <str>}
    """
    logger = get_logger(__name__)

    # Prefer legacy module if present (tests may monkeypatch this name)
    try:
        from functions.llm.client_factory import build_gemini_client  # type: ignore
        legacy = True
    except Exception:
        from functions.llm.client import build_gemini_client  # type: ignore
        legacy = False

    if not legacy:
        try:
            out = build_gemini_client(credentials_cfg, model_name_override=model_name)  # type: ignore[misc]
            if isinstance(out, dict) and "client" in out and "model_name" in out:
                return {"client": out["client"], "model_name": str(out["model_name"])}
            return {"client": out, "model_name": str(model_name)}
        except TypeError as e:
            logger.debug("New client builder signature mismatch, falling back to legacy: %s", str(e))

    client_obj = build_gemini_client(  # type: ignore[misc]
        silence_logs=bool(getattr(params.llm, "silence_client_lv_logs", True))
    )
    return {"client": client_obj, "model_name": str(model_name)}


def _get_thread_client_ctx(params: Any, *, credentials_cfg: Any, model_name: str) -> Dict[str, Any]:
    """
    Ensure each worker thread gets its own client instance unless caller explicitly
    chooses to share. This is safer because many SDK clients are not thread-safe.
    """
    cached = getattr(_CLIENT_TLS, "client_ctx", None)
    if isinstance(cached, dict) and "client" in cached and "model_name" in cached:
        return cached

    bundle = _build_client_bundle(params, credentials_cfg=credentials_cfg, model_name=model_name)
    ctx = {"client": bundle["client"], "model_name": str(bundle["model_name"])}
    _CLIENT_TLS.client_ctx = ctx
    return ctx


# -----------------------------
# Per-item processing
# -----------------------------

def _resolve_context_for_item(item: WorkRef, gc_map: Dict[str, str]) -> str:
    """
    Resolve full context for a WorkItem across modes:

    - Dedup mode: item.group_context_id exists -> lookup in gc_map
    - Non-dedup: item may carry inline context (row-wise / group_output)
    """
    gid = getattr(item, "group_context_id", None)
    if gid:
        ctx = gc_map.get(str(gid))
        if ctx is None:
            raise RuntimeError(
                f"Missing group context id={gid} work_id={item.work_id} group_key={getattr(item, 'group_key', None)}"
            )
        if not str(ctx).strip():
            raise RuntimeError(
                f"Empty group context id={gid} work_id={item.work_id} group_key={getattr(item, 'group_key', None)}"
            )
        return str(ctx)

    inline_ctx = getattr(item, "context", None)
    if isinstance(inline_ctx, str) and inline_ctx.strip():
        return inline_ctx

    raise RuntimeError(
        "WorkItem missing both group_context_id and inline context: "
        f"work_id={item.work_id} group_key={getattr(item, 'group_key', None)}"
    )


def _process_work_item(
    *,
    params: Any,
    credentials_cfg: Any,
    item: WorkRef,
    item_no: int,
    n_total: int,
    gc_map: Dict[str, str],
    cache_id_map: Dict[str, str],
    out_dir: Path,
    fail_dir: Path,
    cache_enabled: bool,
    cache_force: bool,
    gen_prompt_path: Path,
    judge_prompt_path: Optional[Path],
    judge_enabled: bool,
    llm_schema_text: str,
    gen_model: Any,
    judge_model: Optional[Any],
    temperature: float,
    max_retries_outer: int,
    runner_max_retries: int,
    runner_cache_dir: str,
    model_name_configured: str,
    vlog: VerbosityLogger,
) -> WorkItemResult:
    """
    Process a single WorkItem (cache check or LLM generation).

    Thread-safe:
    - Each WorkItem writes to unique files based on cache_id.
    - Each worker thread uses its own client context.
    """
    t_item0 = time.perf_counter()

    context = _resolve_context_for_item(item, gc_map)

    cache_id = cache_id_map.get(item.work_id)
    if not cache_id:
        raise RuntimeError(f"Missing cache_id for work_id={item.work_id}")

    out_file = out_dir / f"{cache_id}.json"

    # Cache hit (double-check; safe even if caller filtered runnable_indices)
    if cache_enabled and (not cache_force) and out_file.exists():
        return WorkItemResult(
            item_no=item_no,
            work_id=item.work_id,
            cache_id=cache_id,
            status="cache_hit",
            output_file=out_file.name,
            elapsed_sec=time.perf_counter() - t_item0,
            error_msg=None,
        )

    # Per-thread client (safer)
    client_ctx = _get_thread_client_ctx(params, credentials_cfg=credentials_cfg, model_name=model_name_configured)
    model_name_effective = str(client_ctx.get("model_name", model_name_configured))

    meta_common: Dict[str, Any] = {
        "cache_id": cache_id,
        "work_id": item.work_id,
        "group_key": getattr(item, "group_key", None),
        "row_index": getattr(item, "row_index", None),
        "group_context_id": getattr(item, "group_context_id", None),
        "model": model_name_effective,
        "temperature": temperature,
        "generation_prompt_path": str(gen_prompt_path),
        "judge_enabled": bool(judge_enabled),
        "judge_prompt_path": str(judge_prompt_path) if judge_prompt_path else None,
        "created_at_utc": _utc_now_iso(),
    }

    res = generate_with_optional_judge(
        context=context,
        llm_schema_text=llm_schema_text,
        gen_prompt_path=gen_prompt_path,
        judge_prompt_path=judge_prompt_path if judge_enabled else None,
        gen_model=gen_model,
        judge_model=judge_model if judge_enabled else None,
        client_ctx=client_ctx,
        temperature=temperature,
        max_retries_outer=max_retries_outer,
        runner_max_retries=runner_max_retries,
        # IMPORTANT: runner cache is NOT the canonical cache. Keep disabled unless explicitly enabled.
        cache_dir=runner_cache_dir,
        runner=run_prompt_yaml_json,
    )

    elapsed = time.perf_counter() - t_item0

    if res.status == "ok" and res.parsed is not None:
        meta_common["attempt_outer"] = int(res.used_attempts)
        write_json(
            out_file,
            {
                "meta": meta_common,
                "parsed": res.parsed.model_dump(),
                "judge": res.judge.model_dump() if res.judge is not None else None,
            },
        )
        return WorkItemResult(
            item_no=item_no,
            work_id=item.work_id,
            cache_id=cache_id,
            status="success",
            output_file=out_file.name,
            elapsed_sec=elapsed,
            error_msg=None,
        )

    # Failure
    meta_common["attempt_outer"] = int(res.used_attempts)
    err = RuntimeError(res.last_error or "LLM generation failed (unknown error)")
    f = write_failure_json(
        failures_dir=fail_dir,
        cache_id=cache_id,
        attempt=int(res.used_attempts),
        meta=meta_common,
        err=err,
    )

    vlog.log(
        0,
        "error",
        "[%d/%d] wid=%s cid=%s FAIL | attempts=%d elapsed=%.2fs | %s | failure=%s",
        item_no,
        n_total,
        item.work_id[:8],
        cache_id[:8],
        int(res.used_attempts),
        elapsed,
        res.last_error or "(no error text)",
        f.name,
    )

    return WorkItemResult(
        item_no=item_no,
        work_id=item.work_id,
        cache_id=cache_id,
        status="failure",
        output_file=f.name,
        elapsed_sec=elapsed,
        error_msg=res.last_error,
    )


# -----------------------------
# Main orchestration
# -----------------------------

def main(
    *,
    parameters_path: str | Path = "configs/parameters.yaml",
    work_items_path: str | Path = PIPELINE3_WORK_ITEMS_PATH_DEFAULT,
    group_contexts_path: str | Path = PIPELINE3_GROUP_CONTEXTS_PATH_DEFAULT,
    outputs_dir: str | Path = LLM_OUTPUTS_DIR_DEFAULT,
    failures_dir: str | Path = LLM_FAILURES_DIR_DEFAULT,
    manifest_path: str | Path = PIPELINE4_MANIFEST_PATH_DEFAULT,
) -> int:
    params = load_parameters(parameters_path)

    configure_logging_from_params(
        params,
        level=str(getattr(getattr(params, "run", None), "log_level", "INFO")),
        log_file=getattr(getattr(params, "run", None), "log_file", None),
    )
    logger = get_logger(__name__)
    _silence_writers_logger()

    # Verbosity is a run concern; keep backward compatible if it lives under cache in older configs.
    verbose = clamp_verbose(
        int(
            getattr(getattr(params, "run", None), "verbose", None)
            or getattr(getattr(params, "cache", None), "verbose", 0)
        )
    )
    vlog = VerbosityLogger(logger, verbose=verbose)

    repo_root = repo_root_from_parameters_path(parameters_path)

    # Resolve key paths deterministically
    gen_prompt_path = resolve_path(params.prompts.generation.path, base_dir=repo_root)
    judge_enabled = bool(getattr(params.prompts.judge, "enabled", False))
    judge_prompt_raw = getattr(params.prompts.judge, "path", None)
    judge_prompt_path = (
        resolve_path(judge_prompt_raw, base_dir=repo_root) if (judge_enabled and judge_prompt_raw) else None
    )

    schema_py_path = resolve_path(params.llm_schema.py_path, base_dir=repo_root)
    schema_txt_path = resolve_path(params.llm_schema.txt_path, base_dir=repo_root)

    # LLM knobs
    model_name = str(params.llm.model_name)
    temperature = float(params.llm.temperature)
    max_retries_outer = int(params.llm.max_retries)
    runner_max_retries = int(getattr(params.llm, "runner_max_retries", params.llm.max_retries))
    max_workers = int(getattr(params.llm, "max_workers", 1))

    # Cache knobs (pipeline-level)
    cache_enabled = bool(params.cache.enabled)
    cache_force = bool(params.cache.force)

    # Runner cache is NOT canonical; default to disabled unless explicitly enabled.
    runner_cache_enabled = bool(getattr(params.cache, "runner_cache_enabled", False))
    runner_cache_dir = str(params.cache.dir) if runner_cache_enabled else ""

    # Load schema.txt as blob
    llm_schema_text = schema_txt_path.read_text(encoding="utf-8")

    # Hash inputs for cache_id stability
    prompt_sha = sha1_file(gen_prompt_path)
    schema_sha = sha1_file(schema_txt_path) if schema_txt_path.exists() else sha1_file(schema_py_path)
    judge_prompt_sha = (
        sha1_file(judge_prompt_path)
        if (judge_enabled and judge_prompt_path and Path(judge_prompt_path).exists())
        else ""
    )

    # Schema models
    gen_model, judge_model = resolve_schema_models(schema_py_path)
    if judge_enabled and (judge_prompt_path is None or judge_model is None):
        raise RuntimeError("Judge enabled but judge prompt path or JudgeResult model missing")

    # Load pipeline 3 artifacts
    wi_obj = read_json(work_items_path)
    items = parse_pipeline3_items(wi_obj)

    # group_contexts artifact may be absent in non-dedup modes; treat missing as empty map
    gc_map: Dict[str, str] = {}
    try:
        gc_obj = read_json(group_contexts_path)
        gc_map = load_group_context_map(gc_obj)
    except FileNotFoundError:
        gc_map = {}

    # Prepare dirs
    out_dir = Path(outputs_dir)
    fail_dir = Path(failures_dir)
    manifest_p = Path(manifest_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)
    manifest_p.parent.mkdir(parents=True, exist_ok=True)

    # Credentials
    credentials_cfg = getattr(params, "credentials", None)
    if credentials_cfg is None:
        cred_path = repo_root / "configs" / "credentials.yaml"
        if cred_path.exists():
            import yaml  # type: ignore
            credentials_cfg = yaml.safe_load(cred_path.read_text(encoding="utf-8"))
        else:
            credentials_cfg = {}

    n_total = len(items)

    vlog.log(
        2,
        "info",
        "Pipeline 4 start | n_items=%d judge=%s cache=%s force=%s model=%s temp=%.3f retries=%d workers=%d verbose=%d",
        n_total,
        bool(judge_enabled),
        bool(cache_enabled),
        bool(cache_force),
        model_name,
        float(temperature),
        max_retries_outer,
        max_workers,
        int(verbose),
    )

    # Pre-scan cache (summary before any LLM run)
    prescan = pre_scan_cache(
        items=items,
        outputs_dir=out_dir,
        cache_enabled=cache_enabled,
        cache_force=cache_force,
        prompt_sha=prompt_sha,
        schema_sha=schema_sha,
        model_name=model_name,
        temperature=temperature,
        judge_enabled=judge_enabled,
        judge_prompt_sha=judge_prompt_sha,
    )

    # Pre-populate cache hits deterministically (so manifest has full success_files list)
    success_files: List[str] = []
    failure_files: List[str] = []

    n_cache_hit = 0
    n_fresh_success = 0
    n_fail = 0

    if cache_enabled and (not cache_force):
        for it in items:
            cid = prescan.cache_id_by_work_id.get(it.work_id)
            if not cid:
                continue
            f = out_dir / f"{cid}.json"
            if f.exists():
                success_files.append(f.name)
                n_cache_hit += 1

    runnable_items: List[WorkRef] = [items[i] for i in prescan.runnable_indices]

    vlog.log(
        2,
        "info",
        "Pipeline 4 pre-scan | cache_skips=%d will_run=%d/%d (cache=%s force=%s) mode=%s",
        prescan.n_cache_skips,
        prescan.n_will_run,
        prescan.n_total,
        bool(cache_enabled),
        bool(cache_force),
        "parallel" if max_workers > 1 else "sequential",
    )

    # Progress cadence
    every_n = item_log_every_n(verbose=verbose)
    if every_n is None:
        every_n = 10

    completed = n_cache_hit  # cache hits count as completed immediately
    t_run0 = time.perf_counter()

    # Sequential execution (max_workers=1)
    if max_workers <= 1:
        for idx, it in enumerate(runnable_items):
            item_no = (n_cache_hit + idx) + 1  # 1-indexed progress position in overall sequence

            result = _process_work_item(
                params=params,
                credentials_cfg=credentials_cfg,
                item=it,
                item_no=item_no,
                n_total=n_total,
                gc_map=gc_map,
                cache_id_map=prescan.cache_id_by_work_id,
                out_dir=out_dir,
                fail_dir=fail_dir,
                cache_enabled=cache_enabled,
                cache_force=cache_force,
                gen_prompt_path=gen_prompt_path,
                judge_prompt_path=judge_prompt_path,
                judge_enabled=judge_enabled,
                llm_schema_text=llm_schema_text,
                gen_model=gen_model,
                judge_model=judge_model,
                temperature=temperature,
                max_retries_outer=max_retries_outer,
                runner_max_retries=runner_max_retries,
                runner_cache_dir=runner_cache_dir,
                model_name_configured=model_name,
                vlog=vlog,
            )

            completed += 1

            if result.status == "cache_hit":
                # Rare here, but safe (race / external file creation)
                n_cache_hit += 1
                if result.output_file:
                    success_files.append(result.output_file)
            elif result.status == "success":
                n_fresh_success += 1
                if result.output_file:
                    success_files.append(result.output_file)
            else:
                n_fail += 1
                if result.output_file:
                    failure_files.append(result.output_file)

            if completed % int(every_n) == 0 or completed == n_total:
                logger.info(
                    "Pipeline 4 progress | %d/%d ok=%d fail=%d cache_hits=%d (pre=%d)",
                    completed,
                    n_total,
                    len(success_files),
                    n_fail,
                    n_cache_hit,
                    prescan.n_cache_skips,
                )

    # Parallel execution (max_workers > 1)
    else:
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(
                    _process_work_item,
                    params=params,
                    credentials_cfg=credentials_cfg,
                    item=it,
                    item_no=0,  # item_no is mostly for log formatting; progress is tracked separately
                    n_total=n_total,
                    gc_map=gc_map,
                    cache_id_map=prescan.cache_id_by_work_id,
                    out_dir=out_dir,
                    fail_dir=fail_dir,
                    cache_enabled=cache_enabled,
                    cache_force=cache_force,
                    gen_prompt_path=gen_prompt_path,
                    judge_prompt_path=judge_prompt_path,
                    judge_enabled=judge_enabled,
                    llm_schema_text=llm_schema_text,
                    gen_model=gen_model,
                    judge_model=judge_model,
                    temperature=temperature,
                    max_retries_outer=max_retries_outer,
                    runner_max_retries=runner_max_retries,
                    runner_cache_dir=runner_cache_dir,
                    model_name_configured=model_name,
                    vlog=vlog,
                ): it
                for it in runnable_items
            }

            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                except Exception as e:
                    with lock:
                        completed += 1
                        n_fail += 1
                    vlog.log(0, "error", "Worker exception: %s", str(e))
                    continue

                with lock:
                    completed += 1

                    if result.status == "cache_hit":
                        n_cache_hit += 1
                        if result.output_file:
                            success_files.append(result.output_file)
                    elif result.status == "success":
                        n_fresh_success += 1
                        if result.output_file:
                            success_files.append(result.output_file)
                    else:
                        n_fail += 1
                        if result.output_file:
                            failure_files.append(result.output_file)

                    if completed % int(every_n) == 0 or completed == n_total:
                        logger.info(
                            "Pipeline 4 progress | %d/%d ok=%d fail=%d cache_hits=%d (pre=%d)",
                            completed,
                            n_total,
                            len(success_files),
                            n_fail,
                            n_cache_hit,
                            prescan.n_cache_skips,
                        )

    elapsed_total = time.perf_counter() - t_run0

    # Backward-compatible counts keys (tests expect n_success)
    n_success = int(len(success_files))
    n_failure = int(len(failure_files))
    n_cache_skipped = int(n_cache_hit)

    manifest = {
        "meta": {
            "pipeline": 4,
            "generated_at_utc": _utc_now_iso(),
            "parameters_path": str(Path(parameters_path)),
            "work_items_path": str(work_items_path),
            "group_contexts_path": str(group_contexts_path),
            "generation_prompt_path": str(gen_prompt_path),
            "judge_enabled": bool(judge_enabled),
            "judge_prompt_path": str(judge_prompt_path) if judge_prompt_path else None,
            "schema_py_path": str(schema_py_path),
            "schema_txt_path": str(schema_txt_path),
            "model_name": model_name,
            "temperature": float(temperature),
            "max_workers": int(max_workers),
            "execution_mode": "parallel" if max_workers > 1 else "sequential",
            "prompt_sha": prompt_sha,
            "schema_sha": schema_sha,
            "judge_prompt_sha": judge_prompt_sha,
            "elapsed_sec": round(elapsed_total, 6),
            "verbose": int(verbose),
            "progress_every_n": int(every_n),
            # Backward-compatible meta keys
            "cache_skips_pre_scan": int(prescan.n_cache_skips),
            "cache_skips_observed": int(n_cache_skipped),
            # Extra diagnostics
            "runner_cache_enabled": bool(runner_cache_enabled),
            "runner_cache_dir": runner_cache_dir or None,
        },
        "counts": {
            # Backward compatible (tests)
            "n_total": int(n_total),
            "n_success": n_success,
            "n_fail": n_failure,
            "n_cache_skipped": n_cache_skipped,
            "n_to_run": int(prescan.n_will_run),
            # Extra breakdown (kept)
            "n_fresh_success": int(n_fresh_success),
            "n_cache_hit": int(n_cache_hit),
        },
        "outputs": {
            "outputs_dir": str(out_dir),
            "failures_dir": str(fail_dir),
            "success_files": sorted(set([s for s in success_files if s])),
            "failure_files": sorted(set([s for s in failure_files if s])),
        },
    }
    write_json(manifest_p, manifest)

    logger.info(
        "Pipeline 4 completed | ok=%d fresh_ok=%d fail=%d cache_skips=%d elapsed=%.2fs mode=%s workers=%d manifest=%s",
        n_success,
        int(n_fresh_success),
        n_failure,
        n_cache_skipped,
        elapsed_total,
        "parallel" if max_workers > 1 else "sequential",
        max_workers,
        str(manifest_p),
    )
    return 0


__all__ = ["main"]
