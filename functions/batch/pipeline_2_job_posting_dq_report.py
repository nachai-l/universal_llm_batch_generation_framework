# functions/batch/pipeline_2_job_posting_dq_report.py
"""
Pipeline 2 — Job Posting DQ Report (Deterministic Aggregation)

Intent
- Read Pipeline 1 output CSV (row-level LLM DQ evaluations with fixed schema).
- Aggregate per-field and overall quality metrics into stakeholder-readable artifacts:
  summary CSVs + Markdown/HTML report.

Core Rules (IMPORTANT)
- Aggregation ALWAYS uses `<FIELD>__status`.
- Anything after `" | "` is ignored for aggregation (status token only).
- Backward compatibility:
  - If Pipeline 1 did NOT output `__status` columns (only raw `"Status | reason"` strings),
    this pipeline will auto-create `__status` columns using a coverage heuristic.
  - Special fields also normalized:
    - `record_validity__status` and `body_readability__status` are derived when missing.

Inputs
- artifacts/reports/job_postings_dq_eval.csv (Pipeline 1)

Outputs (default)
Reports
- artifacts/reports/job_posting_dq_report.md
- artifacts/reports/job_posting_dq_report.html

Core summary tables
- artifacts/reports/job_posting_dq_field_summary.csv
- artifacts/reports/job_posting_dq_overall_summary.csv
- artifacts/reports/job_posting_dq_body_skills_top.csv

P0 additions (high utility)
- Input completeness (structured input availability)
  - artifacts/reports/job_posting_input_completeness.csv
  - Prefers `in__*` columns (Option B style); falls back to canonical input columns.
- Top problem reasons (Unmatch + Unsure)
  - artifacts/reports/job_posting_dq_top_reasons.csv

Enhancements (P1–P2)
1) Field reliability scores
   - reliability_score        = 1 - (unmatch_rate + unsure_rate)
   - reliability_score_strict = 1 - (unmatch_rate + unsure_rate + nodata_rate)
   - artifacts/reports/job_posting_dq_field_reliability.csv

2) Failure mode split (report section)
   - Explicit unmatch/unsure/nodata rates per field in Markdown/HTML report.

3) NoData dominance lens
   - nodata_dominance = nodata_rate / (unmatch_rate + unsure_rate + nodata_rate)
   - artifacts/reports/job_posting_dq_field_nodata_dominance.csv

4) Per-record problem density (record health)
   - problem_fields_count = #fields with {Unmatch, Unsure}
   - problem_fields_rate  = problem_fields_count / #fields with __status
   - artifacts/reports/job_posting_dq_record_health.csv
   - artifacts/reports/job_posting_dq_record_health_distribution.csv

5) BODY skill count per JD
   - Per record:  artifacts/reports/job_posting_dq_body_skill_count_per_record.csv
   - Summary:     artifacts/reports/job_posting_dq_body_skill_count_summary.csv
   - Distribution:artifacts/reports/job_posting_dq_body_skill_count_distribution.csv

6) Title usability (TITLE_NAME vs fallback TITLE_RAW)
   - Table: artifacts/reports/job_posting_dq_title_usability.csv
   - Recommendation:
     "Use TITLE_NAME if TITLE_NAME__status == Match else fallback to TITLE_RAW"

Determinism / Non-LLM Guarantee
- This pipeline does not call any LLM.
- Given the same input CSV, outputs are deterministic.

CLI
- Default:
    python -m functions.batch.pipeline_2_job_posting_dq_report
- With overrides:
    python -m functions.batch.pipeline_2_job_posting_dq_report --input-csv ... --top-n-fields 50
"""


from __future__ import annotations

import argparse
import ast
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Constants
# -----------------------------
ALLOWED_STATUSES = ["Match", "Unmatch", "Unsure", "NoData"]

# These are special fields (not part of ALLOWED_STATUSES)
RECORD_VALIDITY_STATUSES = ["ValidJob", "TestOrSpam", "LowQuality"]
BODY_READABILITY_STATUSES = ["Good", "Fair", "Poor"]

# Columns we should never treat as "evaluated fields"
NON_EVAL_COLS = {
    "ID",
    "URL",
    "body_skills",
    "record_validity",
    "body_readability",
}

# Canonical structured input columns (Pipeline 1 default selection)
# Used as fallback for input completeness if `in__*` columns are absent.
CANONICAL_INPUT_COLS = [
    "BODY",
    "TITLE_RAW",
    "TITLE_NAME",
    "POSTED",
    "EXPIRED",
    "COMPANY_NAME",
    "COMPANY_RAW",
    "COMPANY_IS_STAFFING",
    "EMPLOYMENT_TYPE_NAME",
    "REMOTE_TYPE_NAME",
    "LOT_V7_OCCUPATION_NAME",
    "LOT_V7_SPECIALIZED_OCCUPATION_NAME",
    "ISCED_LEVELS_NAME",
    "NAICS2_NAME",
    "NACE_REVISION2_1_NAME",
    "SALARY_TO",
    "SALARY_FROM",
    "ORIGINAL_PAY_PERIOD",
    "LAA_COUNTRY_NAME",
    "LAA_METRO_NAME",
    "LAA_ADMIN_AREA_1_NAME",
    "LAA_ADMIN_AREA_2_NAME",
]

# Tokens treated as "missing" for input completeness
DEFAULT_MISSING_TOKENS = {
    "",
    "Unclassified",
    "Unclassified Industry",
    "N/A",
    "NA",
    "None",
    "null",
    "NULL",
    "nan",
    "NaN",
    "NoData",
}


# -----------------------------
# Helpers
# -----------------------------
def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def _status_only(x: str) -> str:
    """Extract status token before ' | '."""
    if not isinstance(x, str):
        return ""
    return x.split(" | ", 1)[0].strip()


def _looks_like_allowed_status(x: str, allowed: List[str]) -> bool:
    return _status_only(x) in allowed


def _value_counts_with_expected(series: pd.Series, expected: List[str]) -> Dict[str, int]:
    vc = series.value_counts(dropna=False).to_dict()
    out = {k: int(vc.get(k, 0)) for k in expected}
    out["_Unexpected"] = int(sum(int(v) for k, v in vc.items() if k not in expected))
    return out


def _rate(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


def _as_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s in ("None", "nan", "NaN") else s


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# -----------------------------
# Auto-normalization
# -----------------------------
def _auto_add_status_columns(
    df: pd.DataFrame,
    *,
    allowed_statuses: List[str] = ALLOWED_STATUSES,
    min_coverage: float = 0.50,
) -> None:
    """
    If Pipeline 1 did not output <FIELD>__status columns, but only raw columns like:
      TITLE_NAME = "Match | ..."
    then create TITLE_NAME__status by extracting token before " | ".

    Heuristic:
    - For each candidate column c, if c__status does not exist
    - and >= min_coverage of rows have status token in ALLOWED_STATUSES,
      create c__status.
    """
    n = len(df)
    if n == 0:
        return

    for c in list(df.columns):
        if c in NON_EVAL_COLS:
            continue
        if c.endswith("__status") or c.endswith("__reason"):
            continue

        status_col = f"{c}__status"
        if status_col in df.columns:
            continue

        col = df[c].astype(str)
        ok = col.map(lambda x: _looks_like_allowed_status(x, allowed_statuses))
        coverage = float(ok.sum()) / float(n) if n else 0.0

        if coverage >= float(min_coverage):
            df[status_col] = col.map(_status_only)


def _normalize_special_status_columns(df: pd.DataFrame) -> None:
    """Ensure record_validity__status and body_readability__status exist."""
    if "record_validity" in df.columns and "record_validity__status" not in df.columns:
        df["record_validity__status"] = df["record_validity"].map(_status_only)

    if "body_readability" in df.columns and "body_readability__status" not in df.columns:
        df["body_readability__status"] = df["body_readability"].map(_status_only)


# -----------------------------
# Field detection
# -----------------------------
def _detect_fields(df: pd.DataFrame) -> List[str]:
    """
    Detect evaluated fields based on <FIELD>__status columns.
    Excludes record_validity/body_readability and known non-eval cols.
    """
    fields = set()
    for c in df.columns:
        if not c.endswith("__status"):
            continue
        base = c[: -len("__status")]
        if base in NON_EVAL_COLS:
            continue
        fields.add(base)
    return sorted(fields)


# -----------------------------
# Field summary
# -----------------------------
def build_field_summary(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    n = len(df)
    rows = []

    for f in fields:
        col = f"{f}__status"
        if col not in df.columns:
            continue

        series = df[col].astype(str)
        blank_count = int((series.str.strip() == "").sum())
        counts = _value_counts_with_expected(series, ALLOWED_STATUSES)

        valid_status_count = int(n - blank_count)
        status_coverage_rate = _rate(valid_status_count, n)

        match_rate = _rate(counts["Match"], n)
        unmatch_rate = _rate(counts["Unmatch"], n)
        unsure_rate = _rate(counts["Unsure"], n)
        nodata_rate = _rate(counts["NoData"], n)
        problem_rate = _rate(counts["Unmatch"] + counts["Unsure"], n)

        # (1) Field reliability scores
        reliability_score = _clamp01(1.0 - (unmatch_rate + unsure_rate))
        reliability_score_strict = _clamp01(1.0 - (unmatch_rate + unsure_rate + nodata_rate))

        # (3) NoData dominance among all issues (problem + nodata)
        denom_issues = unmatch_rate + unsure_rate + nodata_rate
        nodata_dominance = (nodata_rate / denom_issues) if denom_issues > 0 else 0.0

        rows.append(
            {
                "field": f,
                "n_rows": n,
                "match_count": counts["Match"],
                "unmatch_count": counts["Unmatch"],
                "unsure_count": counts["Unsure"],
                "nodata_count": counts["NoData"],
                "unexpected_count": counts["_Unexpected"],
                "blank_status_count": blank_count,
                "status_coverage_rate": status_coverage_rate,
                "match_rate": match_rate,
                "unmatch_rate": unmatch_rate,
                "unsure_rate": unsure_rate,
                "nodata_rate": nodata_rate,
                "problem_rate": problem_rate,
                "reliability_score": reliability_score,
                "reliability_score_strict": reliability_score_strict,
                "nodata_dominance": nodata_dominance,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["problem_rate", "field"], ascending=[False, True]).reset_index(drop=True)


# -----------------------------
# Overall summary
# -----------------------------
def build_overall_summary(field_summary: pd.DataFrame) -> pd.DataFrame:
    if field_summary.empty:
        return pd.DataFrame([{"note": "No evaluable fields found"}])

    n_rows = int(field_summary["n_rows"].iloc[0])
    n_fields = int(len(field_summary))

    return pd.DataFrame(
        [
            {
                "n_fields": n_fields,
                "n_rows": n_rows,
                "total_evaluations": int(n_rows * n_fields),
                "total_match": int(field_summary["match_count"].sum()),
                "total_unmatch": int(field_summary["unmatch_count"].sum()),
                "total_unsure": int(field_summary["unsure_count"].sum()),
                "total_nodata": int(field_summary["nodata_count"].sum()),
                "total_unexpected": int(field_summary["unexpected_count"].sum()),
                "total_blank_status": int(field_summary["blank_status_count"].sum()),
                "mean_status_coverage_rate": float(field_summary["status_coverage_rate"].mean()),
                "mean_match_rate_across_fields": float(field_summary["match_rate"].mean()),
                "mean_unmatch_rate_across_fields": float(field_summary["unmatch_rate"].mean()),
                "mean_unsure_rate_across_fields": float(field_summary["unsure_rate"].mean()),
                "mean_nodata_rate_across_fields": float(field_summary["nodata_rate"].mean()),
                "mean_problem_rate_across_fields": float(field_summary["problem_rate"].mean()),
                "mean_reliability_score": float(field_summary["reliability_score"].mean()),
                "mean_reliability_score_strict": float(field_summary["reliability_score_strict"].mean()),
            }
        ]
    )


# -----------------------------
# BODY skills
# -----------------------------
def _try_parse_listlike(x: str) -> List[str]:
    if not isinstance(x, str) or not x.strip():
        return []
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list):
            return [str(i) for i in v if str(i).strip()]
    except Exception:
        pass
    return []


def build_body_skills_top(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if "body_skills" not in df.columns:
        return pd.DataFrame(columns=["skill", "count"])

    skills: List[str] = []
    for cell in df["body_skills"]:
        for s in _try_parse_listlike(str(cell)):
            skills.append(s.split(" | ", 1)[0].strip())  # ignore text after |
    if not skills:
        return pd.DataFrame(columns=["skill", "count"])

    out = pd.Series(skills, dtype=str).value_counts().head(int(top_n)).reset_index()
    out.columns = ["skill", "count"]
    return out


def build_body_skill_count_artifacts(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (5) BODY skill count per JD.

    Returns:
      - per_record: ID, URL, body_skill_count
      - summary: single-row stats (mean/median/pct_zero/p90/p95/max)
      - dist: bins for quick stakeholder read
    """
    n = len(df)
    if n == 0 or "body_skills" not in df.columns:
        per_record = pd.DataFrame(columns=["ID", "URL", "body_skill_count"])
        summary = pd.DataFrame(
            [
                {
                    "n_rows": int(n),
                    "mean": 0.0,
                    "median": 0.0,
                    "pct_zero": 0.0,
                    "p10": 0.0,
                    "p25": 0.0,
                    "p75": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "max": 0.0,
                }
            ]
        )
        dist = pd.DataFrame(columns=["bin", "count", "rate"])
        return per_record, summary, dist

    counts = df["body_skills"].astype(str).map(lambda x: len(_try_parse_listlike(x))).astype(int)

    per_record = pd.DataFrame(
        {
            "ID": df["ID"] if "ID" in df.columns else "",
            "URL": df["URL"] if "URL" in df.columns else "",
            "body_skill_count": counts,
        }
    )

    def _q(p: float) -> float:
        try:
            return float(counts.quantile(p))
        except Exception:
            return 0.0

    pct_zero = float((counts == 0).sum()) / float(n) if n else 0.0

    summary = pd.DataFrame(
        [
            {
                "n_rows": int(n),
                "mean": float(counts.mean()) if n else 0.0,
                "median": float(counts.median()) if n else 0.0,
                "pct_zero": pct_zero,
                "p10": _q(0.10),
                "p25": _q(0.25),
                "p75": _q(0.75),
                "p90": _q(0.90),
                "p95": _q(0.95),
                "max": float(counts.max()) if n else 0.0,
            }
        ]
    )

    # bins: 0, 1–3, 4–7, 8–12, 13–20, 21+
    bins = [
        ("0", lambda x: x == 0),
        ("1–3", lambda x: (x >= 1) & (x <= 3)),
        ("4–7", lambda x: (x >= 4) & (x <= 7)),
        ("8–12", lambda x: (x >= 8) & (x <= 12)),
        ("13–20", lambda x: (x >= 13) & (x <= 20)),
        ("21+", lambda x: x >= 21),
    ]

    rows = []
    for label, fn in bins:
        cnt = int(fn(counts).sum())
        rows.append({"bin": label, "count": cnt, "rate": _rate(cnt, n)})

    dist = pd.DataFrame(rows)
    return per_record, summary, dist


# -----------------------------
# Simple distributions (status-only)
# -----------------------------
def build_status_distribution(df: pd.DataFrame, raw_col: str, expected: List[str]) -> pd.DataFrame:
    """
    Distribution for special fields (record_validity, body_readability):
    - use <raw_col>__status if present, else build it from raw_col
    - ignore everything after " | "
    """
    status_col = f"{raw_col}__status"
    if raw_col in df.columns and status_col not in df.columns:
        df[status_col] = df[raw_col].map(_status_only)

    if status_col not in df.columns:
        return pd.DataFrame(columns=[raw_col, "count", "rate"])

    vc = df[status_col].value_counts(dropna=False)
    n = len(df)

    out = vc.reset_index()
    out.columns = [raw_col, "count"]
    out["rate"] = out["count"].astype(float) / float(n) if n else 0.0
    return out


# -----------------------------
# Input completeness (FIXED)
# -----------------------------
def _detect_input_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Prefer Option-B style columns: `in__*`.
    If none exist, fall back to canonical structured input columns (without prefix).
    Returns (columns, mode_label).
    """
    in_cols = [c for c in df.columns if c.startswith("in__")]
    if in_cols:
        ordered = []
        canonical_pref = [f"in__{c}" for c in CANONICAL_INPUT_COLS]
        for c in canonical_pref:
            if c in in_cols:
                ordered.append(c)
        remaining = sorted([c for c in in_cols if c not in ordered])
        return ordered + remaining, "in__* (Option B)"

    fallback = [c for c in CANONICAL_INPUT_COLS if c in df.columns]
    if fallback:
        return fallback, "canonical inputs (fallback)"

    return [], "no inputs detected"


def build_input_completeness(
    df: pd.DataFrame,
    *,
    input_cols: List[str],
    missing_tokens: Optional[set] = None,
) -> pd.DataFrame:
    """
    For each input column:
    - missing_count = empty string OR token in missing_tokens (case-insensitive match for common NA tokens)
    - valid_count = n_rows - missing_count
    """
    if not input_cols:
        return pd.DataFrame(
            columns=[
                "column",
                "n_rows",
                "missing_count",
                "missing_rate",
                "valid_count",
                "valid_rate",
            ]
        )

    missing_tokens = missing_tokens or set(DEFAULT_MISSING_TOKENS)
    missing_lower = {str(t).strip().lower() for t in missing_tokens}

    n = len(df)
    rows = []
    for c in input_cols:
        s = df[c].astype(str)
        s_stripped = s.map(lambda x: _as_str(x).strip())
        is_empty = s_stripped == ""
        is_token_missing = s_stripped.map(lambda x: x.lower() in missing_lower)

        missing_count = int((is_empty | is_token_missing).sum())
        valid_count = int(n - missing_count)

        rows.append(
            {
                "column": c,
                "n_rows": int(n),
                "missing_count": missing_count,
                "missing_rate": _rate(missing_count, n),
                "valid_count": valid_count,
                "valid_rate": _rate(valid_count, n),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["missing_rate", "column"], ascending=[False, True]).reset_index(drop=True)


# -----------------------------
# Top reasons (Unmatch + Unsure)
# -----------------------------
def build_top_problem_reasons(
    df: pd.DataFrame,
    *,
    fields: List[str],
    top_n_per_field: int = 5,
    max_reason_len: int = 180,
) -> pd.DataFrame:
    """
    For each field:
      - Use <field>__status and <field>__reason
      - Keep statuses in {Unmatch, Unsure}
      - Group by (field, status, reason) -> count
      - Output top N per field (by count desc)

    Returns:
      field, status, reason, count, rate
    """
    n_rows = len(df)
    rows: List[Dict[str, object]] = []

    for f in fields:
        s_col = f"{f}__status"
        r_col = f"{f}__reason"

        if s_col not in df.columns:
            continue

        if r_col not in df.columns:
            if f in df.columns:
                raw = df[f].astype(str)
                df[r_col] = raw.map(lambda x: x.split(" | ", 1)[1].strip() if " | " in x else "")
            else:
                df[r_col] = ""

        tmp = df[[s_col, r_col]].copy()
        tmp[s_col] = tmp[s_col].astype(str).map(lambda x: x.strip())
        tmp[r_col] = tmp[r_col].astype(str).map(lambda x: x.strip())

        tmp = tmp[tmp[s_col].isin(["Unmatch", "Unsure"])].copy()
        if tmp.empty:
            continue

        tmp[r_col] = tmp[r_col].map(lambda x: x[:max_reason_len])

        agg = (
            tmp.groupby([s_col, r_col], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", s_col], ascending=[False, True])
        )

        top = agg.head(int(top_n_per_field))
        for _, row in top.iterrows():
            cnt = int(row["count"])
            rows.append(
                {
                    "field": f,
                    "status": row[s_col],
                    "reason": row[r_col] if row[r_col] else "(no reason)",
                    "count": cnt,
                    "rate": _rate(cnt, n_rows),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["field", "status", "reason", "count", "rate"])

    return out.sort_values(["field", "count", "status"], ascending=[True, False, True]).reset_index(drop=True)


# -----------------------------
# (4) Per-record problem density
# -----------------------------
def build_record_health(
    df: pd.DataFrame,
    *,
    fields: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-record:
      - problem_fields_count: number of fields with status in {Unmatch, Unsure}
      - problem_fields_rate: problem_fields_count / n_fields

    Returns:
      - record_health_df: ID, URL, n_fields, problem_fields_count, problem_fields_rate
      - distribution_df: binned distribution of problem_fields_rate
    """
    n_fields = int(len(fields))
    if n_fields == 0 or len(df) == 0:
        empty_health = pd.DataFrame(columns=["ID", "URL", "n_fields", "problem_fields_count", "problem_fields_rate"])
        empty_dist = pd.DataFrame(columns=["bin", "count", "rate"])
        return empty_health, empty_dist

    status_cols = [f"{f}__status" for f in fields if f"{f}__status" in df.columns]
    if not status_cols:
        empty_health = pd.DataFrame(columns=["ID", "URL", "n_fields", "problem_fields_count", "problem_fields_rate"])
        empty_dist = pd.DataFrame(columns=["bin", "count", "rate"])
        return empty_health, empty_dist

    tmp = df[status_cols].astype(str)
    is_problem = tmp.applymap(lambda x: x.strip() in ("Unmatch", "Unsure"))
    problem_count = is_problem.sum(axis=1).astype(int)

    health = pd.DataFrame(
        {
            "ID": df["ID"] if "ID" in df.columns else "",
            "URL": df["URL"] if "URL" in df.columns else "",
            "n_fields": int(len(status_cols)),
            "problem_fields_count": problem_count,
        }
    )
    denom = float(len(status_cols)) if len(status_cols) else 1.0
    health["problem_fields_rate"] = health["problem_fields_count"].astype(float) / denom

    bins = [
        (0.0, 0.0, "0%"),
        (0.0, 0.1, "0–10%"),
        (0.1, 0.3, "10–30%"),
        (0.3, 0.5, "30–50%"),
        (0.5, 1.0, "50–100%"),
    ]

    n = len(health)
    rows = []
    for lo, hi, label in bins:
        if lo == hi:
            mask = health["problem_fields_rate"] == lo
        else:
            mask = (health["problem_fields_rate"] > lo) & (health["problem_fields_rate"] <= hi)
        cnt = int(mask.sum())
        rows.append({"bin": label, "count": cnt, "rate": _rate(cnt, n)})

    dist = pd.DataFrame(rows)
    return health, dist


# -----------------------------
# (6) Title usability (TITLE_NAME vs fallback TITLE_RAW)
# -----------------------------
def build_title_usability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decide whether TITLE_NAME can be used directly, or should fallback to TITLE_RAW.

    Logic:
      - If TITLE_NAME__status == Match: usable_direct
      - Else: fallback_required (use TITLE_RAW)

    Output table includes:
      n_rows, usable_direct_count/rate, fallback_required_count/rate,
      match/unmatch/unsure/nodata breakdown.
    """
    n = len(df)
    cols = ["TITLE_NAME__status", "TITLE_RAW", "TITLE_NAME"]
    for c in cols:
        if c not in df.columns:
            # still produce a deterministic empty/partial table
            pass

    # Ensure TITLE_NAME__status exists if possible
    if "TITLE_NAME" in df.columns and "TITLE_NAME__status" not in df.columns:
        df["TITLE_NAME__status"] = df["TITLE_NAME"].map(_status_only)

    if "TITLE_NAME__status" not in df.columns or n == 0:
        return pd.DataFrame(
            [
                {
                    "n_rows": int(n),
                    "usable_direct_count": 0,
                    "usable_direct_rate": 0.0,
                    "fallback_required_count": int(n),
                    "fallback_required_rate": 1.0 if n else 0.0,
                    "match_rate": 0.0,
                    "unmatch_rate": 0.0,
                    "unsure_rate": 0.0,
                    "nodata_rate": 0.0,
                    "recommendation": "Use TITLE_NAME if TITLE_NAME__status == Match else fallback to TITLE_RAW",
                }
            ]
        )

    s = df["TITLE_NAME__status"].astype(str).map(lambda x: x.strip())
    match = int((s == "Match").sum())
    unmatch = int((s == "Unmatch").sum())
    unsure = int((s == "Unsure").sum())
    nodata = int((s == "NoData").sum())

    usable_direct = match
    fallback_required = int(n - usable_direct)

    return pd.DataFrame(
        [
            {
                "n_rows": int(n),
                "usable_direct_count": int(usable_direct),
                "usable_direct_rate": _rate(int(usable_direct), n),
                "fallback_required_count": int(fallback_required),
                "fallback_required_rate": _rate(int(fallback_required), n),
                "match_rate": _rate(match, n),
                "unmatch_rate": _rate(unmatch, n),
                "unsure_rate": _rate(unsure, n),
                "nodata_rate": _rate(nodata, n),
                "recommendation": "Use TITLE_NAME if TITLE_NAME__status == Match else fallback to TITLE_RAW",
            }
        ]
    )


# -----------------------------
# Markdown helpers
# -----------------------------
def _to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_(no data)_"
    return df.head(int(max_rows)).to_markdown(index=False)


def _md_to_html(md: str) -> str:
    esc = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<html><body><pre>\n{esc}\n</pre></body></html>"


def _format_rates_for_md(df: pd.DataFrame, cols: List[str], ndp: int = 3) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{float(x):.{ndp}f}" if str(x).strip() != "" else "")
    return out


# -----------------------------
# Main runner
# -----------------------------
def run(
    *,
    input_csv: str = "artifacts/reports/job_postings_dq_eval.csv",
    output_report_md: str = "artifacts/reports/job_posting_dq_report.md",
    output_report_html: str = "artifacts/reports/job_posting_dq_report.html",
    output_field_summary_csv: str = "artifacts/reports/job_posting_dq_field_summary.csv",
    output_overall_summary_csv: str = "artifacts/reports/job_posting_dq_overall_summary.csv",
    output_body_skills_top_csv: str = "artifacts/reports/job_posting_dq_body_skills_top.csv",
    # P0 outputs
    output_input_completeness_csv: str = "artifacts/reports/job_posting_input_completeness.csv",
    output_top_reasons_csv: str = "artifacts/reports/job_posting_dq_top_reasons.csv",
    # Added outputs for 1-4
    output_field_reliability_csv: str = "artifacts/reports/job_posting_dq_field_reliability.csv",
    output_field_nodata_dominance_csv: str = "artifacts/reports/job_posting_dq_field_nodata_dominance.csv",
    output_record_health_csv: str = "artifacts/reports/job_posting_dq_record_health.csv",
    output_record_health_dist_csv: str = "artifacts/reports/job_posting_dq_record_health_distribution.csv",
    # New outputs (5-6)
    output_body_skill_count_per_record_csv: str = "artifacts/reports/job_posting_dq_body_skill_count_per_record.csv",
    output_body_skill_count_summary_csv: str = "artifacts/reports/job_posting_dq_body_skill_count_summary.csv",
    output_body_skill_count_dist_csv: str = "artifacts/reports/job_posting_dq_body_skill_count_distribution.csv",
    output_title_usability_csv: str = "artifacts/reports/job_posting_dq_title_usability.csv",
    # knobs
    top_n_fields: int = 25,
    top_n_skills: int = 50,
    top_n_reasons_per_field: int = 5,
    top_n_input_missing: int = 25,
) -> None:
    df = _safe_read_csv(input_csv)

    # 1) Normalize special fields
    _normalize_special_status_columns(df)

    # 2) Auto-create __status for evaluable fields when missing
    _auto_add_status_columns(df, allowed_statuses=ALLOWED_STATUSES, min_coverage=0.50)

    # 3) Detect + aggregate
    fields = _detect_fields(df)
    field_summary = build_field_summary(df, fields)
    overall_summary = build_overall_summary(field_summary)

    record_validity_dist = build_status_distribution(df, "record_validity", RECORD_VALIDITY_STATUSES)
    body_readability_dist = build_status_distribution(df, "body_readability", BODY_READABILITY_STATUSES)

    body_skills_top = build_body_skills_top(df, top_n_skills)

    # 4) P0 additions: input completeness + top reasons
    input_cols, input_mode = _detect_input_columns(df)
    input_completeness = build_input_completeness(df, input_cols=input_cols)

    top_reasons = build_top_problem_reasons(
        df,
        fields=fields,
        top_n_per_field=int(top_n_reasons_per_field),
        max_reason_len=180,
    )

    # 5) Field reliability + NoData dominance tables
    field_reliability = pd.DataFrame()
    field_nodata_dom = pd.DataFrame()
    if not field_summary.empty:
        field_reliability = field_summary[
            [
                "field",
                "reliability_score",
                "reliability_score_strict",
                "problem_rate",
                "unmatch_rate",
                "unsure_rate",
                "nodata_rate",
            ]
        ].copy()

        field_nodata_dom = (
            field_summary[
                ["field", "nodata_dominance", "nodata_rate", "unmatch_rate", "unsure_rate", "problem_rate"]
            ]
            .copy()
            .sort_values(["nodata_dominance", "nodata_rate", "field"], ascending=[False, False, True])
        )

    # 6) Record health
    record_health, record_health_dist = build_record_health(df, fields=fields)

    # 7) (5) BODY skill count per JD
    body_skill_count_per_record, body_skill_count_summary, body_skill_count_dist = build_body_skill_count_artifacts(df)

    # 8) (6) Title usability
    title_usability = build_title_usability(df)

    # 9) Write CSVs
    _ensure_parent_dir(output_field_summary_csv)
    field_summary.to_csv(output_field_summary_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_overall_summary_csv)
    overall_summary.to_csv(output_overall_summary_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_body_skills_top_csv)
    body_skills_top.to_csv(output_body_skills_top_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_input_completeness_csv)
    input_completeness.to_csv(output_input_completeness_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_top_reasons_csv)
    top_reasons.to_csv(output_top_reasons_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_field_reliability_csv)
    (
        field_reliability
        if not field_reliability.empty
        else pd.DataFrame(
            columns=[
                "field",
                "reliability_score",
                "reliability_score_strict",
                "problem_rate",
                "unmatch_rate",
                "unsure_rate",
                "nodata_rate",
            ]
        )
    ).to_csv(output_field_reliability_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_field_nodata_dominance_csv)
    (
        field_nodata_dom
        if not field_nodata_dom.empty
        else pd.DataFrame(
            columns=["field", "nodata_dominance", "nodata_rate", "unmatch_rate", "unsure_rate", "problem_rate"]
        )
    ).to_csv(output_field_nodata_dominance_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_record_health_csv)
    record_health.to_csv(output_record_health_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_record_health_dist_csv)
    record_health_dist.to_csv(output_record_health_dist_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_body_skill_count_per_record_csv)
    body_skill_count_per_record.to_csv(output_body_skill_count_per_record_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_body_skill_count_summary_csv)
    body_skill_count_summary.to_csv(output_body_skill_count_summary_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_body_skill_count_dist_csv)
    body_skill_count_dist.to_csv(output_body_skill_count_dist_csv, index=False, encoding="utf-8")

    _ensure_parent_dir(output_title_usability_csv)
    title_usability.to_csv(output_title_usability_csv, index=False, encoding="utf-8")

    # 10) Build report tables (formatting for markdown)
    show_field_summary = _format_rates_for_md(
        field_summary,
        cols=[
            "status_coverage_rate",
            "match_rate",
            "unmatch_rate",
            "unsure_rate",
            "nodata_rate",
            "problem_rate",
            "reliability_score",
            "reliability_score_strict",
            "nodata_dominance",
        ],
        ndp=3,
    )

    # (2) Failure mode split
    failure_mode = pd.DataFrame()
    if not field_summary.empty:
        failure_mode = (
            field_summary[["field", "unmatch_rate", "unsure_rate", "nodata_rate", "problem_rate"]]
            .copy()
            .sort_values(["problem_rate", "field"], ascending=[False, True])
        )
        failure_mode = _format_rates_for_md(
            failure_mode,
            cols=["unmatch_rate", "unsure_rate", "nodata_rate", "problem_rate"],
            ndp=3,
        )

    show_input_completeness = _format_rates_for_md(input_completeness, cols=["missing_rate", "valid_rate"], ndp=3)
    show_top_reasons = _format_rates_for_md(top_reasons, cols=["rate"], ndp=3)

    show_nodata_dom = _format_rates_for_md(
        field_nodata_dom, cols=["nodata_dominance", "nodata_rate", "unmatch_rate", "unsure_rate", "problem_rate"], ndp=3
    )

    show_record_health_dist = _format_rates_for_md(record_health_dist, cols=["rate"], ndp=3)

    # Top/bottom reliability
    top5_reliable = pd.DataFrame()
    bottom5_reliable = pd.DataFrame()
    if not field_summary.empty:
        tmp = field_summary[["field", "reliability_score", "reliability_score_strict", "problem_rate"]].copy()
        tmp = tmp.sort_values(["reliability_score", "problem_rate", "field"], ascending=[False, True, True])

        top5_reliable = _format_rates_for_md(
            tmp.head(5),
            cols=["reliability_score", "reliability_score_strict", "problem_rate"],
            ndp=3,
        )
        bottom5_reliable = _format_rates_for_md(
            tmp.tail(5).sort_values(["reliability_score", "problem_rate", "field"], ascending=[True, False, True]),
            cols=["reliability_score", "reliability_score_strict", "problem_rate"],
            ndp=3,
        )

    # (5) Body skill count tables formatting
    show_body_skill_count_summary = _format_rates_for_md(
        body_skill_count_summary, cols=["pct_zero"], ndp=3
    )
    show_body_skill_count_dist = _format_rates_for_md(
        body_skill_count_dist, cols=["rate"], ndp=3
    )

    # (6) Title usability formatting
    show_title_usability = _format_rates_for_md(
        title_usability,
        cols=[
            "usable_direct_rate",
            "fallback_required_rate",
            "match_rate",
            "unmatch_rate",
            "unsure_rate",
            "nodata_rate",
        ],
        ndp=3,
    )

    # 11) Markdown report
    md_lines = [
        "# Job Posting DQ Evaluation — Summary Report",
        "",
        f"- Generated at: `{_now_iso()}`",
        f"- Input CSV: `{input_csv}`",
        "",
        "## Overall Summary",
        _to_md(overall_summary, max_rows=50),
        "",
        "## Record Validity Distribution",
        _to_md(record_validity_dist, max_rows=50),
        "",
        "## Body Readability Distribution",
        _to_md(body_readability_dist, max_rows=50),
        "",
        "## Field Quality Summary (Top problem fields)",
        _to_md(show_field_summary, max_rows=top_n_fields),
        "",
        "## Field Failure Mode Split (Unmatch vs Unsure vs NoData)",
        "_Rates per field (explicit breakdown)._",
        _to_md(failure_mode, max_rows=top_n_fields),
        "",
        "## Field Reliability (Top 5 / Bottom 5)",
        "_reliability_score = 1 - (unmatch + unsure). strict additionally penalizes NoData._",
        "",
        "### Top 5 most reliable fields",
        _to_md(top5_reliable, max_rows=10),
        "",
        "### Bottom 5 least reliable fields",
        _to_md(bottom5_reliable, max_rows=10),
        "",
        "## NoData Dominance (Which fields fail mostly due to missing inputs?)",
        "_nodata_dominance = nodata / (unmatch + unsure + nodata)._",
        _to_md(show_nodata_dom, max_rows=top_n_fields),
        "",
        "## Per-record Problem Density (Unmatch + Unsure across fields)",
        "_Distribution over problem_fields_rate = (#problem fields) / (#fields with __status)._",
        _to_md(show_record_health_dist, max_rows=50),
        "",
        "## Top BODY Skills (frequency)",
        _to_md(body_skills_top, max_rows=top_n_skills),
        "",
        "## BODY Skill Count per JD",
        "_Counts derived from parsed `body_skills` list per record._",
        "",
        "### Summary",
        _to_md(show_body_skill_count_summary, max_rows=10),
        "",
        "### Distribution",
        _to_md(show_body_skill_count_dist, max_rows=50),
        "",
        "## Title Usability (TITLE_NAME vs fallback TITLE_RAW)",
        "_If TITLE_NAME__status != Match, recommend falling back to TITLE_RAW._",
        _to_md(show_title_usability, max_rows=20),
        "",
        "## Input Completeness (Top missing structured fields)",
        f"_Derived from `{input_mode}`. Missing includes empty strings and common Unclassified/NA tokens._",
        _to_md(show_input_completeness, max_rows=top_n_input_missing),
        "",
        "## Top Reasons (Unmatch + Unsure)",
        "_Most frequent reasons per field (truncated for readability)._",
        _to_md(show_top_reasons, max_rows=200),
        "",
        "## Output Artifacts",
        "- Field summary CSV: `artifacts/reports/job_posting_dq_field_summary.csv`",
        "- Overall summary CSV: `artifacts/reports/job_posting_dq_overall_summary.csv`",
        "- BODY skills top CSV: `artifacts/reports/job_posting_dq_body_skills_top.csv`",
        "- BODY skill count per record CSV: `artifacts/reports/job_posting_dq_body_skill_count_per_record.csv`",
        "- BODY skill count summary CSV: `artifacts/reports/job_posting_dq_body_skill_count_summary.csv`",
        "- BODY skill count distribution CSV: `artifacts/reports/job_posting_dq_body_skill_count_distribution.csv`",
        "- Title usability CSV: `artifacts/reports/job_posting_dq_title_usability.csv`",
        "- Input completeness CSV: `artifacts/reports/job_posting_input_completeness.csv`",
        "- Top reasons CSV: `artifacts/reports/job_posting_dq_top_reasons.csv`",
        "- Field reliability CSV: `artifacts/reports/job_posting_dq_field_reliability.csv`",
        "- NoData dominance CSV: `artifacts/reports/job_posting_dq_field_nodata_dominance.csv`",
        "- Record health CSV: `artifacts/reports/job_posting_dq_record_health.csv`",
        "- Record health distribution CSV: `artifacts/reports/job_posting_dq_record_health_distribution.csv`",
    ]
    md_text = "\n".join(md_lines)

    _ensure_parent_dir(output_report_md)
    Path(output_report_md).write_text(md_text, encoding="utf-8")

    _ensure_parent_dir(output_report_html)
    Path(output_report_html).write_text(_md_to_html(md_text), encoding="utf-8")

    print("✅ Pipeline 2 complete")
    print(f"- Report MD:                     {output_report_md}")
    print(f"- Report HTML:                   {output_report_html}")
    print(f"- Field summary CSV:             {output_field_summary_csv}")
    print(f"- Overall summary CSV:           {output_overall_summary_csv}")
    print(f"- BODY skills top CSV:           {output_body_skills_top_csv}")
    print(f"- BODY skill count per record:   {output_body_skill_count_per_record_csv}")
    print(f"- BODY skill count summary:      {output_body_skill_count_summary_csv}")
    print(f"- BODY skill count dist:         {output_body_skill_count_dist_csv}")
    print(f"- Title usability CSV:           {output_title_usability_csv}")
    print(f"- Input completeness CSV:        {output_input_completeness_csv}")
    print(f"- Top reasons CSV:               {output_top_reasons_csv}")
    print(f"- Field reliability CSV:         {output_field_reliability_csv}")
    print(f"- NoData dominance CSV:          {output_field_nodata_dominance_csv}")
    print(f"- Record health CSV:             {output_record_health_csv}")
    print(f"- Record health dist CSV:        {output_record_health_dist_csv}")


# -----------------------------
# CLI
# -----------------------------
def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Pipeline 2: summarize job posting DQ CSV into report + aggregates")
    p.add_argument("--input-csv", default="artifacts/reports/job_postings_dq_eval.csv")
    p.add_argument("--output-report-md", default="artifacts/reports/job_posting_dq_report.md")
    p.add_argument("--output-report-html", default="artifacts/reports/job_posting_dq_report.html")
    p.add_argument("--output-field-summary-csv", default="artifacts/reports/job_posting_dq_field_summary.csv")
    p.add_argument("--output-overall-summary-csv", default="artifacts/reports/job_posting_dq_overall_summary.csv")
    p.add_argument("--output-body-skills-top-csv", default="artifacts/reports/job_posting_dq_body_skills_top.csv")

    # P0 outputs
    p.add_argument("--output-input-completeness-csv", default="artifacts/reports/job_posting_input_completeness.csv")
    p.add_argument("--output-top-reasons-csv", default="artifacts/reports/job_posting_dq_top_reasons.csv")

    # outputs for 1-4
    p.add_argument("--output-field-reliability-csv", default="artifacts/reports/job_posting_dq_field_reliability.csv")
    p.add_argument(
        "--output-field-nodata-dominance-csv",
        default="artifacts/reports/job_posting_dq_field_nodata_dominance.csv",
    )
    p.add_argument("--output-record-health-csv", default="artifacts/reports/job_posting_dq_record_health.csv")
    p.add_argument(
        "--output-record-health-dist-csv",
        default="artifacts/reports/job_posting_dq_record_health_distribution.csv",
    )

    # outputs for 5-6
    p.add_argument(
        "--output-body-skill-count-per-record-csv",
        default="artifacts/reports/job_posting_dq_body_skill_count_per_record.csv",
    )
    p.add_argument(
        "--output-body-skill-count-summary-csv",
        default="artifacts/reports/job_posting_dq_body_skill_count_summary.csv",
    )
    p.add_argument(
        "--output-body-skill-count-dist-csv",
        default="artifacts/reports/job_posting_dq_body_skill_count_distribution.csv",
    )
    p.add_argument(
        "--output-title-usability-csv",
        default="artifacts/reports/job_posting_dq_title_usability.csv",
    )

    # knobs
    p.add_argument("--top-n-fields", type=int, default=25)
    p.add_argument("--top-n-skills", type=int, default=50)
    p.add_argument("--top-n-reasons-per-field", type=int, default=5)
    p.add_argument("--top-n-input-missing", type=int, default=25)

    args = p.parse_args(argv)

    run(
        input_csv=args.input_csv,
        output_report_md=args.output_report_md,
        output_report_html=args.output_report_html,
        output_field_summary_csv=args.output_field_summary_csv,
        output_overall_summary_csv=args.output_overall_summary_csv,
        output_body_skills_top_csv=args.output_body_skills_top_csv,
        output_input_completeness_csv=args.output_input_completeness_csv,
        output_top_reasons_csv=args.output_top_reasons_csv,
        output_field_reliability_csv=args.output_field_reliability_csv,
        output_field_nodata_dominance_csv=args.output_field_nodata_dominance_csv,
        output_record_health_csv=args.output_record_health_csv,
        output_record_health_dist_csv=args.output_record_health_dist_csv,
        output_body_skill_count_per_record_csv=args.output_body_skill_count_per_record_csv,
        output_body_skill_count_summary_csv=args.output_body_skill_count_summary_csv,
        output_body_skill_count_dist_csv=args.output_body_skill_count_dist_csv,
        output_title_usability_csv=args.output_title_usability_csv,
        top_n_fields=args.top_n_fields,
        top_n_skills=args.top_n_skills,
        top_n_reasons_per_field=args.top_n_reasons_per_field,
        top_n_input_missing=args.top_n_input_missing,
    )


if __name__ == "__main__":
    main()
