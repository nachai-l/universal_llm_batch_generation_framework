"""
functions/core/context_builder.py

Intent
------
Build deterministic LLM context strings from a tabular input (DataFrame),
either row-wise or grouped, driven by typed configs (ParametersConfig).

Supports grouping modes (from config.py):
- group_output: one WorkItem per group
- row_output_with_group_context: one WorkItem per row, but with group context

Update (2026-02)
----------------
We add an optional **de-duplication** path for row_output_with_group_context:

- Build group contexts once:
    group_key -> group_context_id -> context
- Each WorkItem stores only:
    group_key + group_context_id
  (not the full repeated context)

This massively reduces artifact size at scale (e.g., 20,000 rows with grouping).

Design principles preserved
---------------------------
- Deterministic outputs (stable ordering, stable hashing)
- No mutation of input df
- Forward-compatible config access (_get)
- Pipeline remains thin; core module contains all logic
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from functions.utils.text import normalize_ws, safe_truncate, to_context_str


# -----------------------------
# Public dataclasses
# -----------------------------

@dataclass(frozen=True)
class WorkItem:
    work_id: str
    group_key: Optional[str]
    row_index: Optional[int]  # original df integer position (iloc index), if applicable
    context: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class GroupContext:
    """
    Represents a de-duplicated group context blob.
    """
    group_context_id: str
    group_key: str
    context: str
    meta: Dict[str, Any]


# -----------------------------
# Public API (legacy-compatible)
# -----------------------------

def build_work_items(df: pd.DataFrame, params: Any) -> List[WorkItem]:
    """
    Build WorkItems from a DataFrame using config-driven rules.

    Parameters
    ----------
    df:
        Input dataframe (must not be mutated).
    params:
        Typed ParametersConfig (Pydantic v2) or any object with similar attributes.

    Returns
    -------
    List[WorkItem]

    Notes
    -----
    This is the original API that returns WorkItems only.
    For row_output_with_group_context, this returns WorkItems with full group context
    repeated per row (legacy behavior).

    For de-dup mode, use:
        build_work_items_and_group_contexts(..., dedupe_group_context=True)
    """
    items, _group_contexts = build_work_items_and_group_contexts(
        df=df,
        params=params,
        dedupe_group_context=False,
    )
    return items


def build_work_items_and_group_contexts(
    df: pd.DataFrame,
    params: Any,
    *,
    dedupe_group_context: bool = False,
) -> Tuple[List[WorkItem], List[GroupContext]]:
    """
    Build WorkItems and (optionally) de-duplicated group contexts.

    When dedupe_group_context=True AND grouping_mode=row_output_with_group_context:
      - returns WorkItems where WorkItem.context is a lightweight placeholder
        (empty by default), and meta includes:
          - group_context_id
      - returns GroupContext list (unique contexts)

    In all other modes:
      - GroupContext list is empty
      - WorkItem.context contains the full computed context (as before)
    """
    if df is None or df.empty:
        return [], []

    # --- grouping config (typed in your config.py) ---
    grouping_enabled = bool(getattr(params.grouping, "enabled", False))
    grouping_column = getattr(params.grouping, "column", None)
    grouping_mode = getattr(params.grouping, "mode", "group_output")
    max_rows_per_group = int(getattr(params.grouping, "max_rows_per_group", 50))

    # --- context config (forward-compatible; uses defaults if missing) ---
    columns_mode = _get(params, "context.columns.mode", "all").strip().lower()
    include_cols = list(_get(params, "context.columns.include", []) or [])
    exclude_cols = list(_get(params, "context.columns.exclude", []) or [])
    row_template = _get(params, "context.row_template", "{__ROW_KV_BLOCK__}") or "{__ROW_KV_BLOCK__}"
    auto_kv_block = bool(_get(params, "context.auto_kv_block", True))
    kv_order = (_get(params, "context.kv_order", "input_order") or "input_order").strip().lower()
    max_context_chars = int(_get(params, "context.max_context_chars", 0) or 0)
    truncate_field_chars = int(_get(params, "context.truncate_field_chars", 0) or 0)

    usable_columns = _select_columns(
        df_columns=list(df.columns),
        mode=columns_mode,
        include=include_cols,
        exclude=exclude_cols,
        kv_order=kv_order,
    )

    # No grouping → row-wise items
    if not grouping_enabled:
        items = _build_rowwise_items(
            df=df,
            usable_columns=usable_columns,
            row_template=row_template,
            auto_kv_block=auto_kv_block,
            truncate_field_chars=truncate_field_chars,
            max_context_chars=max_context_chars,
        )
        return items, []

    # Grouping enabled
    if not grouping_column or grouping_column not in df.columns:
        # Your ParametersConfig already validates this, but keep a safe guard
        raise ValueError(f"grouping.enabled=true but grouping.column not found in df: {grouping_column!r}")

    if grouping_mode == "group_output":
        items = _build_group_output_items(
            df=df,
            grouping_column=grouping_column,
            max_rows_per_group=max_rows_per_group,
            usable_columns=usable_columns,
            row_template=row_template,
            auto_kv_block=auto_kv_block,
            truncate_field_chars=truncate_field_chars,
            max_context_chars=max_context_chars,
        )
        return items, []

    if grouping_mode == "row_output_with_group_context":
        if dedupe_group_context:
            items, group_contexts = _build_row_output_with_group_context_items_deduped(
                df=df,
                grouping_column=grouping_column,
                max_rows_per_group=max_rows_per_group,
                usable_columns=usable_columns,
                row_template=row_template,
                auto_kv_block=auto_kv_block,
                truncate_field_chars=truncate_field_chars,
                max_context_chars=max_context_chars,
            )
            return items, group_contexts

        items = _build_row_output_with_group_context_items(
            df=df,
            grouping_column=grouping_column,
            max_rows_per_group=max_rows_per_group,
            usable_columns=usable_columns,
            row_template=row_template,
            auto_kv_block=auto_kv_block,
            truncate_field_chars=truncate_field_chars,
            max_context_chars=max_context_chars,
        )
        return items, []

    raise ValueError(f"Unsupported grouping.mode: {grouping_mode!r}")


# -----------------------------
# Mode: row-wise (no grouping)
# -----------------------------

def _build_rowwise_items(
    df: pd.DataFrame,
    usable_columns: List[str],
    row_template: str,
    auto_kv_block: bool,
    truncate_field_chars: int,
    max_context_chars: int,
) -> List[WorkItem]:
    items: List[WorkItem] = []
    for iloc_idx in range(len(df)):
        row = df.iloc[iloc_idx]

        kv_block, kv_meta = _row_to_kv_block(row, usable_columns, truncate_field_chars)
        rendered = _render_row(row_template, kv_block, auto_kv_block)

        context, trunc_meta = _truncate_context(rendered, max_context_chars)

        work_id = _stable_id(
            [
                "row",
                str(iloc_idx),
                ",".join(usable_columns),
                kv_block,
                str(truncate_field_chars),
                str(max_context_chars),
            ]
        )

        items.append(
            WorkItem(
                work_id=work_id,
                group_key=None,
                row_index=iloc_idx,
                context=context,
                meta={
                    "mode": "row",
                    "row_iloc_index": iloc_idx,
                    "usable_columns": usable_columns,
                    "kv": kv_meta,
                    "context_truncation": trunc_meta,
                },
            )
        )
    return items


# -----------------------------
# Mode: group_output
# -----------------------------

def _build_group_output_items(
    df: pd.DataFrame,
    grouping_column: str,
    max_rows_per_group: int,
    usable_columns: List[str],
    row_template: str,
    auto_kv_block: bool,
    truncate_field_chars: int,
    max_context_chars: int,
) -> List[WorkItem]:
    items: List[WorkItem] = []

    group_keys = _unique_in_order(df[grouping_column].tolist())

    for gk in group_keys:
        gdf = df[df[grouping_column] == gk]
        row_indices_all = gdf.index.tolist()

        # cap rows per group (iloc cap within filtered frame)
        if max_rows_per_group > 0 and len(gdf) > max_rows_per_group:
            gdf = gdf.iloc[:max_rows_per_group]
            row_cap_applied = True
        else:
            row_cap_applied = False

        row_indices_used = gdf.index.tolist()

        rendered_rows: List[str] = []
        per_row_meta: List[Dict[str, Any]] = []

        for pos in range(len(gdf)):
            row = gdf.iloc[pos]
            kv_block, kv_meta = _row_to_kv_block(row, usable_columns, truncate_field_chars)
            rendered_rows.append(_render_row(row_template, kv_block, auto_kv_block))
            per_row_meta.append({"row_pos_in_group": pos, "kv": kv_meta})

        group_context_raw = "\n\n".join(rendered_rows)
        context, trunc_meta = _truncate_context(group_context_raw, max_context_chars)

        group_context_id = _stable_id(
            [
                "group_context",
                grouping_column,
                str(gk),
                ",".join(usable_columns),
                group_context_raw,
                str(truncate_field_chars),
                str(max_context_chars),
                str(max_rows_per_group),
            ]
        )

        work_id = _stable_id(
            [
                "group_output",
                grouping_column,
                str(gk),
                group_context_id,
            ]
        )

        items.append(
            WorkItem(
                work_id=work_id,
                group_key=str(gk) if gk is not None and not _is_na(gk) else "",
                row_index=None,
                context=context,
                meta={
                    "mode": "group_output",
                    "group_context_id": group_context_id,
                    "grouping_column": grouping_column,
                    "group_key": str(gk) if gk is not None and not _is_na(gk) else "",
                    "usable_columns": usable_columns,
                    "row_cap_applied": row_cap_applied,
                    "row_indices_all": row_indices_all,
                    "row_indices_used": row_indices_used,
                    "per_row": per_row_meta,
                    "context_truncation": trunc_meta,
                },
            )
        )

    return items


# -----------------------------
# Mode: row_output_with_group_context (legacy behavior)
# -----------------------------

def _build_row_output_with_group_context_items(
    df: pd.DataFrame,
    grouping_column: str,
    max_rows_per_group: int,
    usable_columns: List[str],
    row_template: str,
    auto_kv_block: bool,
    truncate_field_chars: int,
    max_context_chars: int,
) -> List[WorkItem]:
    """
    One WorkItem per row, but each WorkItem's context is the whole group context.
    This is useful when each output is row-level, but needs group-level shared context.

    Determinism:
    - group context is built in input order for that group (stable)
    - work_id includes row identity + full group context
    """
    items: List[WorkItem] = []

    # Precompute group contexts once (deterministic + efficient)
    group_context_map: Dict[str, Dict[str, Any]] = {}

    group_keys = _unique_in_order(df[grouping_column].tolist())
    for gk in group_keys:
        gdf = df[df[grouping_column] == gk]

        row_indices_all = gdf.index.tolist()
        if max_rows_per_group > 0 and len(gdf) > max_rows_per_group:
            gdf_ctx = gdf.iloc[:max_rows_per_group]
            row_cap_applied = True
        else:
            gdf_ctx = gdf
            row_cap_applied = False

        row_indices_used = gdf_ctx.index.tolist()

        rendered_rows: List[str] = []
        for pos in range(len(gdf_ctx)):
            row = gdf_ctx.iloc[pos]
            kv_block, _ = _row_to_kv_block(row, usable_columns, truncate_field_chars)
            rendered_rows.append(_render_row(row_template, kv_block, auto_kv_block))

        group_context_raw = "\n\n".join(rendered_rows)
        context, trunc_meta = _truncate_context(group_context_raw, max_context_chars)

        gk_str = str(gk) if gk is not None and not _is_na(gk) else ""
        group_context_map[gk_str] = {
            "context": context,
            "raw_group_context": group_context_raw,
            "row_cap_applied": row_cap_applied,
            "row_indices_all": row_indices_all,
            "row_indices_used": row_indices_used,
            "context_truncation": trunc_meta,
        }

    # Now create per-row items with that group's context
    for iloc_idx in range(len(df)):
        gk_val = df.iloc[iloc_idx][grouping_column]
        gk_str = str(gk_val) if gk_val is not None and not _is_na(gk_val) else ""
        ginfo = group_context_map.get(gk_str)
        if ginfo is None:
            # should not happen, but safe guard
            continue

        context = ginfo["context"]

        # ID includes the row + group context (so caching invalidates if group changes)
        work_id = _stable_id(
            [
                "row_with_group_context",
                str(iloc_idx),
                grouping_column,
                gk_str,
                ",".join(usable_columns),
                ginfo["raw_group_context"],
                str(truncate_field_chars),
                str(max_context_chars),
                str(max_rows_per_group),
            ]
        )

        items.append(
            WorkItem(
                work_id=work_id,
                group_key=gk_str,
                row_index=iloc_idx,
                context=context,
                meta={
                    "mode": "row_output_with_group_context",
                    "row_iloc_index": iloc_idx,
                    "grouping_column": grouping_column,
                    "group_key": gk_str,
                    "usable_columns": usable_columns,
                    "row_cap_applied": ginfo["row_cap_applied"],
                    "row_indices_all": ginfo["row_indices_all"],
                    "row_indices_used": ginfo["row_indices_used"],
                    "context_truncation": ginfo["context_truncation"],
                },
            )
        )

    return items


# -----------------------------
# Mode: row_output_with_group_context (DEDUPED)
# -----------------------------

def _build_row_output_with_group_context_items_deduped(
    df: pd.DataFrame,
    grouping_column: str,
    max_rows_per_group: int,
    usable_columns: List[str],
    row_template: str,
    auto_kv_block: bool,
    truncate_field_chars: int,
    max_context_chars: int,
) -> Tuple[List[WorkItem], List[GroupContext]]:
    """
    De-duplicated version:
      - Build each group context once, assign stable group_context_id
      - Each row WorkItem references group_context_id in meta
      - WorkItem.context is left empty (or can be a small placeholder)

    Determinism:
      - group order: first-seen in input order
      - row order: df iloc order
      - group_context_id includes the full *raw group context* so it changes if group changes
    """
    items: List[WorkItem] = []
    group_contexts: List[GroupContext] = []

    group_keys = _unique_in_order(df[grouping_column].tolist())

    # Build contexts once
    group_map: Dict[str, Dict[str, Any]] = {}
    for gk in group_keys:
        gdf = df[df[grouping_column] == gk]

        row_indices_all = gdf.index.tolist()
        if max_rows_per_group > 0 and len(gdf) > max_rows_per_group:
            gdf_ctx = gdf.iloc[:max_rows_per_group]
            row_cap_applied = True
        else:
            gdf_ctx = gdf
            row_cap_applied = False

        row_indices_used = gdf_ctx.index.tolist()

        rendered_rows: List[str] = []
        for pos in range(len(gdf_ctx)):
            row = gdf_ctx.iloc[pos]
            kv_block, _ = _row_to_kv_block(row, usable_columns, truncate_field_chars)
            rendered_rows.append(_render_row(row_template, kv_block, auto_kv_block))

        raw_group_context = "\n\n".join(rendered_rows)
        context, trunc_meta = _truncate_context(raw_group_context, max_context_chars)

        gk_str = str(gk) if gk is not None and not _is_na(gk) else ""

        group_context_id = _stable_id(
            [
                "group_context",
                grouping_column,
                gk_str,
                ",".join(usable_columns),
                raw_group_context,
                str(truncate_field_chars),
                str(max_context_chars),
                str(max_rows_per_group),
            ]
        )

        gc = GroupContext(
            group_context_id=group_context_id,
            group_key=gk_str,
            context=context,
            meta={
                "grouping_column": grouping_column,
                "group_key": gk_str,
                "usable_columns": usable_columns,
                "row_cap_applied": row_cap_applied,
                "row_indices_all": row_indices_all,
                "row_indices_used": row_indices_used,
                "context_truncation": trunc_meta,
                "raw_group_context_len": len(raw_group_context),
            },
        )
        group_contexts.append(gc)

        group_map[gk_str] = {
            "group_context_id": group_context_id,
            "row_cap_applied": row_cap_applied,
            "row_indices_all": row_indices_all,
            "row_indices_used": row_indices_used,
            "context_truncation": trunc_meta,
            "raw_group_context": raw_group_context,
        }

    # Per-row items referencing the group context id
    for iloc_idx in range(len(df)):
        gk_val = df.iloc[iloc_idx][grouping_column]
        gk_str = str(gk_val) if gk_val is not None and not _is_na(gk_val) else ""
        ginfo = group_map.get(gk_str)
        if ginfo is None:
            continue

        # Work id must invalidate if group context changes, so include the raw group context in the id.
        work_id = _stable_id(
            [
                "row_with_group_context_ref",
                str(iloc_idx),
                grouping_column,
                gk_str,
                ginfo["group_context_id"],
                ",".join(usable_columns),
                ginfo["raw_group_context"],
                str(truncate_field_chars),
                str(max_context_chars),
                str(max_rows_per_group),
            ]
        )

        items.append(
            WorkItem(
                work_id=work_id,
                group_key=gk_str,
                row_index=iloc_idx,
                context="",  # intentionally empty; prompt builder will join with group context artifact
                meta={
                    "mode": "row_output_with_group_context",
                    "deduped_group_context": True,
                    "row_iloc_index": iloc_idx,
                    "grouping_column": grouping_column,
                    "group_key": gk_str,
                    "group_context_id": ginfo["group_context_id"],
                    "usable_columns": usable_columns,
                    "row_cap_applied": ginfo["row_cap_applied"],
                    "row_indices_all": ginfo["row_indices_all"],
                    "row_indices_used": ginfo["row_indices_used"],
                    "context_truncation": ginfo["context_truncation"],
                },
            )
        )

    return items, group_contexts


# -----------------------------
# Column selection
# -----------------------------

def _select_columns(
    df_columns: List[str],
    mode: str,
    include: List[str],
    exclude: List[str],
    kv_order: str,
) -> List[str]:
    mode = (mode or "all").strip().lower()
    include = [c for c in include if c in df_columns]
    exclude_set = {c for c in exclude if c in df_columns}

    if mode == "include":
        cols = [c for c in include if c not in exclude_set]
    elif mode == "exclude":
        cols = [c for c in df_columns if c not in exclude_set]
    else:  # all
        cols = [c for c in df_columns if c not in exclude_set]

    if kv_order == "alpha":
        cols = sorted(cols)

    return cols


# -----------------------------
# KV building + rendering
# -----------------------------

def _row_to_kv_block(
    row: pd.Series,
    usable_columns: List[str],
    truncate_field_chars: int,
) -> Tuple[str, Dict[str, Any]]:
    lines: List[str] = []
    field_truncated: Dict[str, bool] = {}

    for col in usable_columns:
        val = row[col] if col in row.index else None
        sval = normalize_ws(to_context_str(val))

        if truncate_field_chars and truncate_field_chars > 0:
            sval, applied = safe_truncate(sval, truncate_field_chars)
            field_truncated[col] = applied
        else:
            field_truncated[col] = False

        lines.append(f"{col}: {sval}")

    kv_block = "\n".join(lines).rstrip()
    meta = {
        "field_truncated": field_truncated,
        "truncate_field_chars": truncate_field_chars,
        "n_fields": len(usable_columns),
    }
    return kv_block, meta


def _render_row(row_template: str, kv_block: str, auto_kv_block: bool) -> str:
    # Currently same behavior for both branches; keep param for forward-compat
    return row_template.replace("{__ROW_KV_BLOCK__}", kv_block)


# -----------------------------
# Truncation
# -----------------------------

def _truncate_context(text: str, max_chars: int) -> Tuple[str, Dict[str, Any]]:
    if not max_chars or max_chars <= 0:
        return text, {
            "applied": False,
            "max_chars": max_chars,
            "original_len": len(text),
            "final_len": len(text),
        }

    truncated, applied = safe_truncate(text, max_chars)
    return truncated, {
        "applied": applied,
        "max_chars": max_chars,
        "original_len": len(text),
        "final_len": len(truncated),
    }


# -----------------------------
# Utilities
# -----------------------------

def _get(obj: Any, dotted_path: str, default: Any = None) -> Any:
    """
    Safe nested getter supporting Pydantic models and dicts.
    Example: _get(params, "context.columns.mode", "all")
    """
    cur: Any = obj
    for part in dotted_path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
            continue
        if hasattr(cur, part):
            cur = getattr(cur, part)
            continue
        return default
    return cur if cur is not None else default


def _stable_id(parts: Sequence[str]) -> str:
    blob = "\n".join([p if p is not None else "" for p in parts]).encode("utf-8", errors="replace")
    return sha1(blob).hexdigest()


def _is_na(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _unique_in_order(seq: Sequence[Any]) -> List[Any]:
    """
    Deterministic unique preserving first-seen order.
    Treats NA-like values as the same bucket.
    """
    seen = set()
    out: List[Any] = []
    for x in seq:
        key = "__NA__" if _is_na(x) else x
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


__all__ = [
    "WorkItem",
    "GroupContext",
    "build_work_items",
    "build_work_items_and_group_contexts",
]
