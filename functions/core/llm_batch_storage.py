# functions/core/llm_batch_storage.py
"""
LLM Batch Storage — deterministic cache identity + pre-scan + failure persistence.

Intent
- Provide reusable, pipeline-agnostic helpers for:
  - stable cache id derivation
  - cache pre-scan (how many will skip vs run)
  - writing failure JSON artifacts

Why this exists
- Keeps pipelines thin (orchestration only).
- Keeps cache + artifacts behavior consistent across pipelines/jobs.

Notes
- Does NOT read params.yaml directly.
- Does NOT depend on pipeline_4 naming.

Compatibility
- The cache_id prefix is intentionally set to "pipeline4" to remain backward-compatible
  with existing on-disk caches in artifacts/cache/llm_outputs created by the earlier
  Pipeline 4 implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from functions.utils.hashing import sha1_text


# -----------------------------
# Cache identity
# -----------------------------

def stable_cache_id(
    *,
    work_id: str,
    prompt_sha: str,
    schema_sha: str,
    model_name: str,
    temperature: float,
    judge_enabled: bool,
    judge_prompt_sha: str = "",
) -> str:
    """
    Deterministic cache key for the final accepted output (post-judge if enabled).

    IMPORTANT:
    - Changing any of the inputs will change the cache_id.
    - Keep the blob stable and ordered.
    - "pipeline4" prefix is used for backward compatibility with existing cache files.
    """
    blob = "\n".join(
        [
            "pipeline4",  # backward-compatible cache namespace
            str(work_id),
            str(prompt_sha),
            str(schema_sha),
            str(model_name),
            f"temp={float(temperature):.6f}",
            f"judge={bool(judge_enabled)}",
            str(judge_prompt_sha or ""),
        ]
    )
    return sha1_text(blob)


def short_id(s: str, n: int = 8) -> str:
    return str(s)[:n]


# -----------------------------
# Cache pre-scan
# -----------------------------

@dataclass(frozen=True)
class CachePreScan:
    """
    Result of pre-scanning items against the output cache directory.
    """
    n_total: int
    n_cache_skips: int
    n_will_run: int
    # work_id -> cache_id mapping (useful for logging)
    cache_id_by_work_id: Dict[str, str]
    # indices of items that should be executed (cache miss or force)
    runnable_indices: list[int]


def pre_scan_cache(
    *,
    items: Sequence[Any],
    outputs_dir: str | Path,
    cache_enabled: bool,
    cache_force: bool,
    prompt_sha: str,
    schema_sha: str,
    model_name: str,
    temperature: float,
    judge_enabled: bool,
    judge_prompt_sha: str = "",
) -> CachePreScan:
    """
    Pre-compute cache_ids and determine which items will run.

    Expected item shape:
      - item.work_id: str

    We intentionally check cache against the "final accepted outputs" directory
    (e.g. artifacts/cache/llm_outputs/{cache_id}.json).
    """
    out_dir = Path(outputs_dir)
    cache_id_by_work_id: Dict[str, str] = {}
    runnable: list[int] = []
    n_skips = 0

    for idx, it in enumerate(items):
        wid = str(getattr(it, "work_id", ""))
        if not wid:
            # Keep this strict; caller should validate items before calling.
            raise ValueError(f"Item missing work_id at index={idx}")

        cid = stable_cache_id(
            work_id=wid,
            prompt_sha=prompt_sha,
            schema_sha=schema_sha,
            model_name=model_name,
            temperature=temperature,
            judge_enabled=judge_enabled,
            judge_prompt_sha=judge_prompt_sha,
        )
        cache_id_by_work_id[wid] = cid

        out_file = out_dir / f"{cid}.json"
        is_hit = cache_enabled and (not cache_force) and out_file.exists()

        if is_hit:
            n_skips += 1
        else:
            runnable.append(idx)

    n_total = len(items)
    n_will_run = n_total - n_skips
    return CachePreScan(
        n_total=n_total,
        n_cache_skips=n_skips,
        n_will_run=n_will_run,
        cache_id_by_work_id=cache_id_by_work_id,
        runnable_indices=runnable,
    )


# -----------------------------
# Failure artifacts
# -----------------------------

def write_failure_json(
    failures_dir: str | Path,
    *,
    cache_id: str,
    attempt: int,
    meta: Mapping[str, Any],
    err: Exception,
) -> Path:
    """
    Write a stable failure artifact:
      {failures_dir}/{cache_id}__a{attempt}.json
    """
    p = Path(failures_dir) / f"{cache_id}__a{attempt}.json"
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": dict(meta),
        "error": {"type": type(err).__name__, "message": str(err)},
    }
    p.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return p


__all__ = [
    "stable_cache_id",
    "short_id",
    "CachePreScan",
    "pre_scan_cache",
    "write_failure_json",
]
