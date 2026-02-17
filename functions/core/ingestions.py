# functions/core/ingestions.py
"""
functions.core.ingestions

Universal delimited ingestion + deterministic cleaning helpers.

Includes:
- read_delimited_table(): robust CSV/TSV/PSV reader (multiline quoted fields supported)
- clean_dataframe(): deterministic cell normalization for PSV conversion and downstream
- clean_delimited_to_psv(): read + clean + write stable PSV
- ingest_input_table_pipeline2(): Pipeline 2 ingestion -> readable JSON artifact payload

Design decisions (Pipeline 2)
- Data types: read as str for determinism
- Missing cells: normalize missing/blank -> "NaN"
- Artifact format: JSON (readable)

Note
- clean_dataframe() is intended for PSV export / normalization use cases and follows
  the expectations in tests/test_ingestions.py (null-ish tokens become "").
- ingest_input_table_pipeline2() is intended for readable inspection artifacts and uses
  "NaN" sentinel for missing cells (per pipeline 2 design decision).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import pandas as pd

InputFormat = Literal["csv", "tsv", "psv"]

_DEFAULT_DELIMS: Dict[str, str] = {
    "csv": ",",
    "tsv": "\t",
    "psv": "|",
}

# Common "null-ish" strings found in exports (case-insensitive)
_NULL_TOKENS = {
    "",
    "none",
    "[none]",
    "null",
    "nan",
    "na",
    "n/a",
    "<na>",
}

def _normalize_cell(x: object) -> str:
    """
    Normalize a single cell to a cleaned string (PSV-friendly).

    Rules (deterministic, traceability-safe):
    - None / pandas NA -> ""
    - Trim outer whitespace
    - Replace CR/LF/TAB with single spaces (so each record stays one line in PSV)
    - Normalize common null tokens (e.g., "NULL", "n/a", "[None]") to ""
    - DO NOT strip backslashes globally (preserve original content)
    """
    if x is None:
        return ""

    if pd.isna(x):
        return ""

    s = str(x)

    # Normalize line structure
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    s = s.replace("\n", " ")

    # Normalize whitespace deterministically
    s = " ".join(s.split()).strip()

    # Normalize null-like tokens (case-insensitive)
    if s.lower() in _NULL_TOKENS:
        return ""

    return s


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame deterministically:
    - preserve columns as-is
    - normalize every cell to a cleaned string (PSV-friendly)
    """
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_normalize_cell)
    return out


def read_delimited_table(
    path: str | Path,
    fmt: InputFormat = "csv",
    *,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Read a delimited file robustly:
    - handles quoted multi-line fields via engine="python"
    - skips malformed rows instead of failing the pipeline
    - dtype=str, keep_default_na=False to avoid implicit NaN conversion

    NOTE:
    - This returns the raw read result (newlines inside quoted fields are preserved).
      clean_dataframe() flattens those newlines later.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    fmt_norm = str(fmt).lower().strip()
    if fmt_norm not in _DEFAULT_DELIMS:
        raise ValueError(f"Unsupported fmt: {fmt}. Expected one of: csv|tsv|psv")

    sep = delimiter if delimiter is not None else _DEFAULT_DELIMS[fmt_norm]

    df = pd.read_csv(
        p,
        sep=sep,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        engine="python",
        on_bad_lines="skip",
    )
    return df


def clean_delimited_to_psv(
    input_file: str | Path,
    output_file: str | Path,
    *,
    fmt: InputFormat = "csv",
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Read a delimited table, clean it, and write deterministic PSV.

    PSV output behavior:
    - sep='|'
    - QUOTE_NONE
    - escapechar='\\'
    - utf-8
    """
    df = read_delimited_table(input_file, fmt=fmt, delimiter=delimiter, encoding=encoding)
    df_clean = clean_dataframe(df)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(
        out_path,
        sep="|",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )

    return df_clean


# ---------------------------------------------------------------------
# Pipeline 2 ingestion (readable JSON artifact)
# ---------------------------------------------------------------------

def _validate_required_columns_present(df: pd.DataFrame, required: Optional[Iterable[str]]) -> None:
    req = [c for c in (required or []) if str(c).strip()]
    if not req:
        return
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")


def _read_input_table_any(
    path: str | Path,
    fmt: str,
    *,
    encoding: str = "utf-8",
    sheet: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pipeline-2 oriented reader:
    - reads everything as str deterministically
    - keep_default_na=False (avoid pandas NaN)
    - supports csv/tsv/psv/xlsx
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    fmt_norm = str(fmt).lower().strip()

    if fmt_norm in ("csv", "tsv", "psv"):
        sep = _DEFAULT_DELIMS.get(fmt_norm)
        if sep is None:
            raise ValueError(f"Unsupported fmt: {fmt}. Expected one of: csv|tsv|psv|xlsx")
        return pd.read_csv(p, sep=sep, encoding=encoding, dtype=str, keep_default_na=False)

    if fmt_norm == "xlsx":
        sheet_name = sheet if sheet is not None else "Sheet1"
        return pd.read_excel(p, sheet_name=sheet_name, dtype=str, keep_default_na=False)

    raise ValueError(f"Unsupported fmt: {fmt}. Expected one of: csv|tsv|psv|xlsx")


def ingest_input_table_pipeline2(params: Any) -> dict[str, Any]:
    """
    Pipeline 2 core ingestion.

    Expected by tests/test_pipeline_2_ingest_input.py:
      - obj["meta"]["n_rows"] exists
      - obj["meta"]["n_cols"] exists
      - obj["rows"] exists (list of row dicts)
    """
    in_cfg = params.input
    df = _read_input_table_any(
        in_cfg.path,
        in_cfg.format,
        encoding=getattr(in_cfg, "encoding", "utf-8"),
        sheet=getattr(in_cfg, "sheet", None),
    )

    _validate_required_columns_present(df, getattr(in_cfg, "required_columns", None))

    # Normalize missing/blank -> "NaN" sentinel (for readable artifact)
    df2 = df.copy()
    for col in df2.columns:
        s = df2[col].astype(str)
        s = s.map(lambda v: "NaN" if (v is None or str(v).strip() == "" or pd.isna(v)) else str(v))
        df2[col] = s

    rows = df2.to_dict(orient="records")

    n_rows = int(df2.shape[0])
    n_cols = int(df2.shape[1])
    cols = list(df2.columns)

    payload: dict[str, Any] = {
        "meta": {
            "input_path": str(in_cfg.path),
            "input_format": str(in_cfg.format),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": cols,  # ✅ add this
        },
        "columns": cols,  # (optional keep)
        "rows": rows,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "records": rows,
    }

    return payload


__all__ = [
    "clean_dataframe",
    "read_delimited_table",
    "clean_delimited_to_psv",
    "ingest_input_table_pipeline2",
]
