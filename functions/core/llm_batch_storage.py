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

New in 2026-02:
- Optional context-sensitive cache identity:
  - stable_cache_id(..., context_sha=...) can incorporate a resolved context hash.
  - pre_scan_cache(..., context_resolver=callable) will compute context_sha per item.
  - If context_resolver is omitted, behavior remains work_id-based (backward compatible).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

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
    # NEW: optional context hash (recommended for GROUP_OUTPUT / grouped contexts)
    context_sha: str = "",
) -> str:
    """
    Deterministic cache key for the final accepted output (post-judge if enabled).

    IMPORTANT:
    - Changing any of the inputs will change the cache_id.
    - Keep the blob stable and ordered.
    - "pipeline4" prefix is used for backward compatibility with existing cache files.

    Context-sensitive caching:
    - If context_sha is provided (non-empty), it becomes part of the identity.
    - This prevents incorrect cache hits when the same work_id is reused but context changes.
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
            # Keep last so older implementations can be conceptually compared
            str(context_sha or ""),
        ]
    )
    return sha1_text(blob)


def context_to_sha(context: str) -> str:
    """
    Stable hash for a resolved context string.
    """
    return sha1_text(str(context))


def short_id(s: str, n: int = 8) -> str:
    return str(s)[:n]


# -----------------------------
# Cache pre-scan
# -----------------------------

ContextResolver = Callable[[Any], str]


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
    # NEW: optional resolver to make cache context-sensitive
    context_resolver: Optional[ContextResolver] = None,
) -> CachePreScan:
    """
    Pre-compute cache_ids and determine which items will run.

    Expected item shape:
      - item.work_id: str

    We intentionally check cache against the "final accepted outputs" directory
    (e.g. artifacts/cache/llm_outputs/{cache_id}.json).

    Context-sensitive caching:
    - If context_resolver is provided, we compute context_sha per item and include it
      into the cache_id. This prevents stale hits when a work_id remains stable but
      the resolved group context changes.
    """
    out_dir = Path(outputs_dir)
    cache_id_by_work_id: Dict[str, str] = {}
    runnable: list[int] = []
    n_skips = 0

    for idx, it in enumerate(items):
        wid = str(getattr(it, "work_id", "")).strip()
        if not wid:
            raise ValueError(f"Item missing work_id at index={idx}")

        ctx_sha = ""
        if context_resolver is not None:
            # Fail early if context is not resolvable; better than silently creating a wrong cache key.
            ctx_text = context_resolver(it)
            if not isinstance(ctx_text, str) or not ctx_text.strip():
                raise ValueError(f"Resolved context is empty for work_id={wid} index={idx}")
            ctx_sha = context_to_sha(ctx_text)

        cid = stable_cache_id(
            work_id=wid,
            prompt_sha=prompt_sha,
            schema_sha=schema_sha,
            model_name=model_name,
            temperature=temperature,
            judge_enabled=judge_enabled,
            judge_prompt_sha=judge_prompt_sha,
            context_sha=ctx_sha,
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
    "context_to_sha",
    "short_id",
    "CachePreScan",
    "pre_scan_cache",
    "write_failure_json",
]
