# functions/core/processing.py
"""
functions.core.processing

Deterministic, LLM-agnostic processing helpers (task-agnostic).

Includes:
- clean_string_columns():
    Normalize whitespace and common unicode artifacts across string columns.
    Returns (df_clean, stats_df). No printing by default.
- row_to_json():
    Extract a record by row index as dict or pretty JSON.
- row_to_json_by_id():
    Extract a record by ID lookup (exact or substring match) as dict or pretty JSON.

Design principles
- Pure pandas + stdlib.
- Deterministic transforms.
- Debug-friendly but pipeline-safe (no noisy prints unless explicitly requested).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import pandas as pd


# -------------------------
# String normalization
# -------------------------

# Match all whitespace runs (includes tabs/newlines etc.)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

# Known problematic unicode spaces / zero-width chars
_UNICODE_REPLACEMENTS = {
    "\u00A0": " ",   # NBSP
    "\u2007": " ",   # figure space
    "\u202F": " ",   # narrow NBSP
    "\u200B": "",    # zero-width space
    "\u200C": "",    # zero-width non-joiner
    "\u200D": "",    # zero-width joiner
    "\uFEFF": "",    # BOM / zero-width no-break
}


def _clean_text_value(x: Any) -> Any:
    """
    Normalize a single value if it is a string; otherwise return as-is.

    Rules:
    - Convert CR/LF/TAB to spaces
    - Replace problematic unicode spaces/chars
    - Collapse whitespace runs to single spaces
    - Strip edges
    """
    if not isinstance(x, str):
        return x

    s = x

    # Normalize newlines/tabs
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")

    # Unicode fixes
    for k, v in _UNICODE_REPLACEMENTS.items():
        s = s.replace(k, v)

    # Collapse whitespace
    s = _WS_RE.sub(" ", s).strip()

    return s


def clean_string_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    *,
    inplace: bool = False,
    return_stats: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Clean string columns by normalizing whitespace and unicode artifacts.

    Args:
        df: input DataFrame
        columns: explicit columns to clean (default: all object/string columns)
        inplace: if True, mutate df; else return a copy
        return_stats: if True, return a stats DataFrame (unique before/after)

    Returns:
        (df_clean, stats_df_or_None)
    """
    out = df if inplace else df.copy()

    if columns is None:
        # include both object and pandas string dtype
        cols = list(out.select_dtypes(include=["object", "string"]).columns)
    else:
        cols = [c for c in columns if c in out.columns]

    stats_rows = []
    for col in cols:
        before = out[col].nunique(dropna=False)
        out[col] = out[col].map(_clean_text_value)
        after = out[col].nunique(dropna=False)

        stats_rows.append(
            {
                "column": col,
                "unique_before": int(before),
                "unique_after": int(after),
                "unique_reduced": int(before - after),
                "reduction_pct": float((before - after) / before * 100.0) if before else 0.0,
            }
        )

    stats_df = pd.DataFrame(stats_rows) if return_stats else None
    return out, stats_df


# -------------------------
# Row extraction helpers
# -------------------------

_NULL_HANDLING = {"keep", "remove", "empty_string"}


def row_to_json(
    df: pd.DataFrame,
    row_idx: int,
    *,
    exclude_cols: Optional[Iterable[str]] = None,
    pretty: bool = True,
    null_handling: str = "keep",
) -> str | dict:
    """
    Convert a specific row from DataFrame to JSON using row index.
    """
    if exclude_cols is None:
        exclude_cols = []

    if null_handling not in _NULL_HANDLING:
        raise ValueError(f"null_handling must be one of {_NULL_HANDLING}")

    if row_idx < 0 or row_idx >= len(df):
        raise ValueError(f"Row index {row_idx} out of bounds (valid range: 0-{len(df)-1})")

    row = df.iloc[row_idx]

    row_dict: dict[str, Any] = {}
    for col in df.columns:
        if col in exclude_cols:
            continue

        value = row[col]
        if pd.isna(value):
            if null_handling == "remove":
                continue
            if null_handling == "empty_string":
                value = ""
            else:
                value = None

        row_dict[str(col)] = value

    return json.dumps(row_dict, indent=2, ensure_ascii=False) if pretty else row_dict


def row_to_json_by_id(
    df: pd.DataFrame,
    record_id: str,
    *,
    id_col: str = "ID",
    exclude_cols: Optional[Iterable[str]] = None,
    pretty: bool = True,
    null_handling: str = "keep",
    strip_id: bool = True,
    allow_partial_match: bool = False,
) -> str | dict:
    """
    Convert a specific row from DataFrame to JSON using an ID lookup.
    """
    if exclude_cols is None:
        exclude_cols = []

    if null_handling not in _NULL_HANDLING:
        raise ValueError(f"null_handling must be one of {_NULL_HANDLING}")

    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in df.columns")

    if record_id is None or str(record_id).strip() == "":
        raise ValueError("record_id is empty")

    rid = str(record_id).strip() if strip_id else str(record_id)

    s = df[id_col].astype(str)
    if strip_id:
        s = s.str.strip()

    if allow_partial_match:
        mask = s.str.contains(rid, na=False)
    else:
        mask = s == rid

    hits = df[mask]
    if hits.empty:
        example_ids = df[id_col].astype(str).head(5).tolist()
        raise ValueError(
            f"record_id '{rid}' not found in column '{id_col}'. Example IDs: {example_ids}"
        )

    if len(hits) > 1:
        raise ValueError(
            f"record_id '{rid}' matched {len(hits)} rows in '{id_col}'. "
            "Please deduplicate or provide a more specific key."
        )

    row = hits.iloc[0]

    row_dict: dict[str, Any] = {}
    for col in df.columns:
        if col in exclude_cols:
            continue

        value = row[col]
        if pd.isna(value):
            if null_handling == "remove":
                continue
            if null_handling == "empty_string":
                value = ""
            else:
                value = None

        row_dict[str(col)] = value

    return json.dumps(row_dict, indent=2, ensure_ascii=False) if pretty else row_dict


__all__ = [
    "clean_string_columns",
    "row_to_json",
    "row_to_json_by_id",
]
