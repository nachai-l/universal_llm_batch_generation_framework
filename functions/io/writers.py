# functions/io/writers.py
"""
Writers (Deterministic Artifacts I/O)

Intent
- Provide a single, deterministic way to write pipeline outputs to disk.
- Standardize how artifacts are persisted across pipelines:
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
- write_jsonl(path, records) -> None
- write_csv(path, df) -> None

Key behaviors / guarantees
- **Deterministic output**
  - JSONL:
    - One JSON object per line
    - UTF-8 encoding
    - sort_keys=True to ensure stable key ordering
    - ensure_ascii=False to preserve Thai / Unicode text
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
    - number of rows written
    - number of columns (CSV)

Design notes
- Writers are intentionally *thin*:
  - No schema enforcement
  - No column mutation
  - No validation logic
- Schema correctness and ordering are responsibilities of upstream pipeline stages.
- This module is reusable across:
  - batch pipelines
  - report generation
  - intermediate debugging artifacts

Typical usage
- JSONL: write per-record LLM outputs for traceability and replay
- CSV: write flattened, fixed-schema outputs for downstream pipelines
"""


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

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


def write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """
    Write records to JSONL deterministically:
    - UTF-8
    - One JSON object per line
    - sort_keys=True for stable output
    - ensure_ascii=False to preserve Thai text
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


__all__ = ["ensure_parent_dir", "write_jsonl", "write_csv"]
