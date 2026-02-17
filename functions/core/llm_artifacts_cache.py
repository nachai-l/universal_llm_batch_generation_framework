# functions/core/llm_artifacts_cache.py
"""
Pipeline 4 Artifact Helpers (Pipeline3 adapters)

Intent
- Centralize Pipeline 3 artifact parsing:
  - pipeline3_work_items.json -> list of WorkRefs
  - pipeline3_group_contexts.json -> group_context_id -> context mapping

Note: stable_cache_id() is in llm_batch_storage.py (cache layer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class WorkRef:
    work_id: str
    group_key: str
    row_index: Optional[int]
    group_context_id: Optional[str]
    context: Optional[str]
    meta: Dict[str, Any]


def parse_pipeline3_items(obj: Dict[str, Any]) -> List[WorkRef]:
    """
    Parse pipeline3_work_items.json into a list of WorkRef.

    Supported shapes (per-item):
      - {"work_id": "...", "group_key": "...", "row_index": null|int, "group_context_id": "...", "context": "...", "meta": {...}}
      - legacy/alt: group_context_id / group_key / context may be under meta
    """
    items = obj.get("items", [])
    if not isinstance(items, list):
        raise ValueError("pipeline3_work_items.json malformed: items is not a list")

    out: List[WorkRef] = []
    for it in items:
        if not isinstance(it, dict):
            continue

        meta = dict(it.get("meta") or {})

        # forward/backward compatible field resolution
        group_context_id = it.get("group_context_id") or meta.get("group_context_id")
        group_key = it.get("group_key") or meta.get("group_key") or ""

        # Inline context for non-dedup modes (e.g., group_output or row-wise inline)
        # Prefer top-level; fall back to meta for forward compatibility.
        inline_context = it.get("context")
        if inline_context is None:
            inline_context = meta.get("context")

        # Normalize types
        work_id = str(it.get("work_id", "") or "").strip()
        if not work_id:
            # Skip malformed entries rather than creating unusable WorkRefs
            continue

        row_index_val = it.get("row_index")
        row_index = int(row_index_val) if row_index_val is not None else None

        ctx: Optional[str] = None
        if isinstance(inline_context, str) and inline_context.strip():
            ctx = inline_context

        gid: Optional[str] = None
        if group_context_id:
            gid_str = str(group_context_id).strip()
            gid = gid_str if gid_str else None

        out.append(
            WorkRef(
                work_id=work_id,
                group_key=str(group_key or ""),
                row_index=row_index,
                group_context_id=gid,
                context=ctx,
                meta=meta,
            )
        )

    return out


def _load_from_groups_list(groups: List[Any]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for g in groups:
        if not isinstance(g, dict):
            continue
        gid = g.get("group_context_id")
        ctx = g.get("context")
        if isinstance(gid, str) and isinstance(ctx, str):
            m[gid] = ctx
    if not m:
        raise ValueError("pipeline3_group_contexts.json malformed: groups list has no valid entries")
    return m


def load_group_context_map(obj: Any) -> Dict[str, str]:
    """
    Build group_context_id -> context mapping from pipeline3_group_contexts.json.

    Supports:
      A) {"groups":[{"group_context_id":"...", "context":"..."}]}
      B) {"<group_context_id>": {"context":"..."}}
      C) [{"group_context_id":"...", "context":"..."}]
    """
    # C) top-level list style
    if isinstance(obj, list):
        return _load_from_groups_list(obj)

    if not isinstance(obj, dict):
        raise ValueError("pipeline3_group_contexts.json malformed: expected dict or list")

    # B) direct mapping style
    m: Dict[str, str] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, dict) and isinstance(v.get("context"), str):
            m[k] = v["context"]
    if m:
        return m

    # A) groups list style
    groups = obj.get("groups")
    if isinstance(groups, list):
        return _load_from_groups_list(groups)

    raise ValueError("pipeline3_group_contexts.json malformed: cannot resolve context mapping")


__all__ = [
    "WorkRef",
    "parse_pipeline3_items",
    "load_group_context_map",
]
