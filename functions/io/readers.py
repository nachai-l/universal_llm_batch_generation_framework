# functions/io/readers.py
"""
Reader (Unified CSV/TSV/PSV/XLSX)

Intent
- Provide a unified reader for multiple input formats, driven by parameters.yaml:
  - csv / tsv / psv
  - xlsx with configurable sheet (default: "sheet1")
- Validate required columns (typical CLO pipeline expectations):
  - course_name_th, course_name_en, clo_name

External calls
- pandas.read_csv / pandas.read_excel
- functions.utils.text.trim_lr (used for trimming column headers only)

Primary functions
- read_input_table(path, fmt, sheet_name=None, encoding="utf-8") -> pandas.DataFrame
- validate_required_columns(df, required_columns) -> None (raise ValueError if missing)

Key behaviors / guarantees
- **File existence check**: raises FileNotFoundError if the input file path does not exist.
- **Supported formats**: csv, tsv, psv, xlsx (case-insensitive).
- **Delimiter mapping**:
  - csv -> ","
  - tsv -> "\\t"
  - psv -> "|"
- **No cell-value trimming**:
  - This reader *intentionally* does not trim cell values to preserve traceability.
  - Only column names are trimmed defensively.
- **Column name normalization**:
  - _trim_column_names() applies trim_lr() to each column header.
  - Helps when input headers contain leading/trailing whitespace.

Validation
- validate_required_columns(df, required_columns):
  - Computes missing columns by exact name match against df.columns.
  - Raises ValueError with both missing and found columns for debugging.

XLSX specifics
- Uses pandas.read_excel with:
  - dtype=str
  - keep_default_na=False (so blanks stay as empty strings, not NaN)
- Default sheet name is **"sheet1"** unless overridden by sheet_name.
  - Note: in many Excel files the default sheet is "Sheet1" (capital S).
    This implementation is strict; if the sheet is actually "Sheet1" you must pass it explicitly.

Error handling / failure modes
- Unsupported fmt -> ValueError listing expected formats.
- Missing required columns -> ValueError including found columns.
- Nonexistent file -> FileNotFoundError.

Where this fits in the pipeline
- Reader-level responsibility: “load table + normalize headers”.
- Pipeline_0 responsibility: “trim/clean values, deduplicate, canonicalize, apply parameters.yaml rules”.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Literal, Optional

import pandas as pd

from functions.utils.text import trim_lr

InputFormat = Literal["csv", "tsv", "psv", "xlsx"]


_DELIMS = {
    "csv": ",",
    "tsv": "\t",
    "psv": "|",
}


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """
    Raise ValueError if any required column is missing.
    """
    required = list(required_columns)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")


def _trim_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive: strip whitespace around column names only.
    """
    df = df.copy()
    df.columns = [trim_lr(str(c)) for c in df.columns]
    return df


def read_input_table(
    path: str | Path,
    fmt: InputFormat,
    sheet_name: Optional[str] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Read an input table from csv/tsv/psv/xlsx.

    Notes:
    - This function DOES NOT trim cell values (to preserve traceability).
      Value trimming/dedup occurs in pipeline_0 according to parameters.yaml.
    - Column names are trimmed defensively.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {str(p)}")

    fmt = fmt.lower().strip()  # type: ignore[assignment]
    if fmt in ("csv", "tsv", "psv"):
        delim = _DELIMS[fmt]  # type: ignore[index]
        df = pd.read_csv(p, sep=delim, encoding=encoding, dtype=str, keep_default_na=False)
        return _trim_column_names(df)

    if fmt == "xlsx":
        sheet = sheet_name or "sheet1"
        df = pd.read_excel(p, sheet_name=sheet, dtype=str, keep_default_na=False)
        return _trim_column_names(df)

    raise ValueError(f"Unsupported input format: {fmt}. Expected one of: csv|tsv|psv|xlsx")
