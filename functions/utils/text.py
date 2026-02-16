"""
Text utilities (small, dependency-free)

Intent
- Keep tiny helpers that are shared across IO / processing modules.
- Avoid trimming cell values (traceability); only use for column headers unless explicitly needed.
"""

from __future__ import annotations

import math
from typing import Any, Tuple


def trim_lr(s: str) -> str:
    """
    Trim leading/trailing whitespace (left+right).
    Keep internal whitespace unchanged.
    """
    return str(s).strip()


def normalize_ws(s: str) -> str:
    """
    Normalize whitespace deterministically:
    - Trim ends
    - Collapse internal whitespace (spaces/newlines/tabs) into single spaces

    Use ONLY for derived strings (context building), not for mutating source cells.
    """
    return " ".join(str(s).split()).strip()


def safe_truncate(s: str, max_chars: int, ellipsis: str = "…") -> Tuple[str, bool]:
    """
    Truncate a string to max_chars with a trailing ellipsis (default: '…').

    Returns (result, applied).
    If max_chars <= 0, returns (s, False).
    """
    if not max_chars or max_chars <= 0:
        return s, False
    if len(s) <= max_chars:
        return s, False
    return s[:max_chars].rstrip() + ellipsis, True


def to_context_str(val: Any) -> str:
    """
    Convert arbitrary values into a context-safe string:
    - None -> ""
    - float NaN -> ""
    - everything else -> str(val)

    Note: dependency-free (no pandas). Handles float NaN via math.isnan.
    """
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val)
