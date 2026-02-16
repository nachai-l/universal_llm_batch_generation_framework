# functions/io/writers.py
"""
Writers (Deterministic Artifacts I/O)

Intent
- Provide a single, deterministic way to write pipeline outputs to disk.
- Standardize how artifacts are persisted across pipelines:
  - JSON (readable, deterministic)
  - JSONL for record-level outputs (LLM results, bundles, traceability)
  - CSV for tabular metrics and reports
- Centralize directory creation and logging for all write operations.

External calls
- json.dumps
- pandas.DataFrame.to_csv
- pathlib.Path
- functions.utils.logging.get_logger

Primary functions
- ensure_parent_dir(path) -> None
- write_json(path, obj) -> None
- write_jsonl(path, records) -> None
- write_csv(path, df) -> None

Key behaviors / guarantees
- **Deterministic output**
  - JSON:
    - UTF-8 encoding
    - sort_keys=True to ensure stable key ordering
    - ensure_ascii=False to preserve Unicode text
    - pretty indent (readability)
  - JSONL:
    - One JSON object per line
    - UTF-8 encoding
    - sort_keys=True to ensure stable key ordering
    - ensure_ascii=False to preserve Unicode text
  - CSV:
    - UTF-8 encoding
    - index=False
    - Column order is exactly df.columns (caller-controlled)

- **Filesystem safety**
  - ensure_parent_dir() is called automatically before writing.
  - Parent directories are created recursively and idempotently.
  - Safe to call repeatedly across pipeline stages.

- **Observability**
  - All write operations emit INFO-level logs with:
    - file path
    - bytes written (JSON)
    - number of rows written (JSONL)
    - number of rows/columns (CSV)

Design notes
- Writers are intentionally *thin*:
  - No schema enforcement
  - No column mutation
  - No validation logic
- Schema correctness and ordering are responsibilities of upstream pipeline stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from functions.utils.logging import get_logger


def ensure_parent_dir(path: str | Path) -> None:
    """
    Ensure parent directory exists for the given file path.
    Idempotent and safe.
    """
    p = Path(path)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def write_json(
    path: str | Path,
    obj: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    """
    Write an object to JSON deterministically.

    Notes:
    - Caller must ensure `obj` is JSON-serializable (dict/list/str/num/bool/None),
      or provide a pre-converted representation (e.g., df.to_dict(orient="records")).
    """
    logger = get_logger(__name__)
    ensure_parent_dir(path)

    p = Path(path)
    text = json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, indent=indent)
    p.write_text(text + "\n", encoding="utf-8")

    # Use len(text) for determinism even on some filesystems; still log actual size if available.
    try:
        size = p.stat().st_size
    except OSError:
        size = len((text + "\n").encode("utf-8"))

    logger.info("Wrote JSON: %s (bytes=%d)", str(p), int(size))


def write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """
    Write records to JSONL deterministically:
    - UTF-8
    - One JSON object per line
    - sort_keys=True for stable output
    - ensure_ascii=False to preserve Unicode text
    """
    logger = get_logger(__name__)
    ensure_parent_dir(path)

    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
            f.write(line + "\n")

    logger.info("Wrote JSONL: %s (rows=%d)", str(p), len(records))


def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    """
    Write DataFrame to CSV deterministically:
    - UTF-8
    - index=False
    - stable column order as df.columns
    """
    logger = get_logger(__name__)
    ensure_parent_dir(path)

    p = Path(path)
    df.to_csv(p, index=False, encoding="utf-8")

    logger.info("Wrote CSV: %s (rows=%d, cols=%d)", str(p), int(df.shape[0]), int(df.shape[1]))


__all__ = ["ensure_parent_dir", "write_json", "write_jsonl", "write_csv"]
