# functions/batch/pipeline_1_job_posting_dq_eval.py
"""
Pipeline 1 — Job Posting Data Quality Evaluation (Batch LLM) + Fixed-Schema Report

Intent
- Evaluate the quality and consistency of Lightcast job posting fields by comparing
  structured fields against raw evidence (BODY, TITLE_RAW, COMPANY_RAW) using Gemini.
- Produce a deterministic, downstream-friendly CSV report with a **fixed schema**
  that Pipeline 2 can aggregate without schema drift.

Inputs
- Raw Lightcast exports (CSV):
  - raw_data/Thailand_global_postings.csv
  - raw_data/Thailand_global_raw.csv
  - raw_data/Thailand_global_skills.csv (optional; cleaned but not used further here)
- Parameters:
  - configs/parameters.yaml (runtime + model + outputs + concurrency)
  - configs/credentials.yaml (non-secret client config; API key comes from ENV)

Preprocessing (deterministic, non-LLM)
1) Clean CSV → PSV (robust handling of quoted multiline / escaped artifacts)
2) Robust whitespace normalization across string columns
3) Merge raw_jds + postings (RIGHT join on ID)
4) Select a stable subset of columns used for evaluation

LLM Evaluation
- For each record, call `run_prompt_json()` using `prompt_key` (default: job_posting_dq_eval_v1).
- Uses per-record caching (`cache_id = "{ID}__{prompt_key}"`) to avoid repeat calls.
- Supports concurrency via ThreadPoolExecutor (configurable `llm.max_workers`).
- Thread-local Gemini client is used to avoid cross-thread reuse issues.

Outputs (Artifacts)
1) JSONL (traceability): `artifacts/reports/job_postings_dq_eval.jsonl`
   - Contains the filtered LLM output + ID (+ URL when present)
   - Keeps the per-record result for auditing and debugging

2) CSV (fixed schema): `artifacts/reports/job_postings_dq_eval.csv`
   - **Stable column order and column set** (no dynamic `in__*`, no surprise columns)
   - For each evaluated field F:
       - F (raw "Status | reason" string from LLM)
       - F__status (parsed status)
       - F__reason (parsed reason)
   - Special handling:
       - body_readability and record_validity behave like normal fields (status/reason)
       - body_skills is stored as a JSON string list (no __status/__reason)

Key Guarantees (IMPORTANT)
- The output CSV schema is always identical across runs:
  - Missing fields are emitted as blank strings
  - Extra LLM keys are ignored (e.g., POST_DATE, ISCED_LEVEL_NAME, etc.)
- Ordering is deterministic:
  - Outputs are sorted by ID before writing JSONL and CSV
- Pipeline 2 compatibility:
  - Pipeline 2 can rely on `EVAL_FIELDS_IN_ORDER` + fixed columns without defensive logic

CLI Usage
- Run default:
    python -m functions.batch.pipeline_1_job_posting_dq_eval
- With overrides:
    python -m functions.batch.pipeline_1_job_posting_dq_eval --max-rows 500 --force
    python -m functions.batch.pipeline_1_job_posting_dq_eval --prompt-key job_posting_dq_eval_v2

Notes / Assumptions
- Secrets are NOT stored in repo. Set ENV:
    export GEMINI_API_KEY="..."
- If `llm.model_name` is set in parameters.yaml, it is mapped into GEMINI_MODEL
  unless GEMINI_MODEL is already present in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict

import functions.core.ingestions as ingest
import functions.core.processing as processing
from functions.llm.client import build_gemini_client
from functions.llm.runner import run_prompt_json
from functions.utils.config import ensure_dirs, load_credentials, load_parameters
from functions.utils.logging import configure_logging, get_logger


class AnyJsonObject(BaseModel):
    """Accept any keys returned by LLM (we will filter later)."""
    model_config = ConfigDict(extra="allow")


# -----------------------------
# Fixed schema (THIS is the CSV structure you want)
# -----------------------------
EVAL_FIELDS_IN_ORDER: List[str] = [
    "COMPANY_IS_STAFFING",
    "COMPANY_NAME",
    "EMPLOYMENT_TYPE_NAME",
    "EXPIRED",
    "ISCED_LEVELS_NAME",
    "LAA_ADMIN_AREA_1_NAME",
    "LAA_ADMIN_AREA_2_NAME",
    "LAA_COUNTRY_NAME",
    "LAA_METRO_NAME",
    "LOT_V7_OCCUPATION_NAME",
    "LOT_V7_SPECIALIZED_OCCUPATION_NAME",
    "NACE_REVISION2_1_NAME",
    "NAICS2_NAME",
    "ORIGINAL_PAY_PERIOD",
    "POSTED",
    "REMOTE_TYPE_NAME",
    "SALARY_FROM",
    "SALARY_TO",
    "TITLE_NAME",
]

SPECIAL_FIELDS = ["body_readability", "record_validity"]  # also status/reason
BODY_SKILLS_FIELD = "body_skills"  # JSON list string, no status/reason


def _expected_csv_columns() -> List[str]:
    cols = ["ID", "URL"]

    # Main evaluated fields
    for f in EVAL_FIELDS_IN_ORDER:
        cols.extend([f, f"{f}__status", f"{f}__reason"])

    # body_readability (special)
    f = "body_readability"
    cols.extend([f, f"{f}__status", f"{f}__reason"])

    # body_skills MUST be here (before record_validity)
    cols.append("body_skills")

    # record_validity (special)
    f = "record_validity"
    cols.extend([f, f"{f}__status", f"{f}__reason"])

    return cols



# -----------------------------
# Config helpers
# -----------------------------
def _get_progress_every(params: Any) -> int:
    default = 100
    llm = getattr(params, "llm", None)
    v = getattr(llm, "progress_log_every", None) if llm else None
    if isinstance(v, int) and v > 0:
        return v

    run_mode = getattr(params, "run_mode", None)
    v = getattr(run_mode, "progress_log_every", None) if run_mode else None
    if isinstance(v, int) and v > 0:
        return v

    return default


def _resolve_row_limit(params: Any, override: Optional[str | int]) -> Optional[int]:
    def _parse(v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("all", ""):
                return None
            try:
                n = int(s)
            except Exception:
                return None
            return n if n > 0 else None
        if isinstance(v, int):
            return v if v > 0 else None
        return None

    if override is not None:
        return _parse(override)

    llm = getattr(params, "llm", None)
    n = _parse(getattr(llm, "max_rows_per_run", None) if llm else None)
    if n is not None:
        return n

    run_mode = getattr(params, "run_mode", None)
    return _parse(getattr(run_mode, "max_rows_per_run", None) if run_mode else None)


def _get_max_workers(params: Any) -> int:
    llm = getattr(params, "llm", None)
    v = getattr(llm, "max_workers", None) if llm else None
    try:
        n = int(v) if v is not None else 1
    except Exception:
        n = 1
    return max(1, n)


def _cache_id(record_id: str, *, prompt_key: str) -> str:
    return f"{record_id}__{prompt_key}"


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v)
    if s.lower() in ("nan", "none"):
        return ""
    return s


# -----------------------------
# Data preprocessing (your exact flow)
# -----------------------------
def _build_post_w_jd_df(
    *,
    raw_postings_csv: str,
    raw_jds_csv: str,
    raw_skills_csv: Optional[str],
    processed_postings_psv: str,
    processed_raw_psv: str,
    processed_skills_psv: Optional[str],
    cols: Optional[List[str]] = None,
    logger=None,
) -> pd.DataFrame:
    if logger:
        logger.info("Preprocess: clean CSV -> PSV")
    postings_df = ingest.clean_csv_to_psv_pandas(raw_postings_csv, processed_postings_psv)
    raw_jds_df = ingest.clean_csv_to_psv_pandas(raw_jds_csv, processed_raw_psv)

    if raw_skills_csv and processed_skills_psv:
        _ = ingest.clean_csv_to_psv_pandas(raw_skills_csv, processed_skills_psv)

    if logger:
        logger.info("Preprocess: robust string cleaning (postings)")
    postings_df, _ = processing.clean_string_columns_robust(postings_df, inplace=True)

    if logger:
        logger.info("Preprocess: robust string cleaning (raw_jds)")
    raw_jds_df, _ = processing.clean_string_columns_robust(raw_jds_df, inplace=True)

    if raw_skills_csv and processed_skills_psv:
        if logger:
            logger.info("Preprocess: robust string cleaning (skills)")
        skills_df = pd.read_csv(processed_skills_psv, sep="|", dtype=str, keep_default_na=False)
        skills_df, _ = processing.clean_string_columns_robust(skills_df, inplace=True)
        # skills_df not used further in this pipeline

    default_cols = [
        "ID",
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
        "URL",
    ]
    use_cols = cols or default_cols

    if logger:
        logger.info("Preprocess: merge raw_jds_df + postings_df (how=right) on ID, select cols")
    post_w_jd_df = raw_jds_df.merge(postings_df, on=["ID"], how="right")

    for c in use_cols:
        if c not in post_w_jd_df.columns:
            post_w_jd_df[c] = ""

    return post_w_jd_df[use_cols].copy()


# -----------------------------
# CSV report flattening
# -----------------------------
def _split_status_reason(v: Any) -> Tuple[str, str]:
    """
    Input format: "<Status> | <reason>"
    Return: (status, reason). If not parsable, (raw, "").
    """
    s = _as_str(v).strip()
    if not s:
        return "", ""
    if " | " in s:
        parts = s.split(" | ", 1)
        return parts[0].strip(), parts[1].strip()
    return s, ""


def _normalize_body_skills(v: Any) -> List[str]:
    """
    Force body_skills into list[str] with minimal cleanup.
    Accepts list[str] or JSON-ish string.
    Returns [] on failure.
    """
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            s = _as_str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        # Try JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
        # Try python literal
        try:
            import ast
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    return []


def _filter_llm_output_to_schema(llm_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only the keys we care about (fixed schema).
    Drop surprise keys (POST_DATE, ISCED_LEVEL_NAME, etc).
    """
    keep_keys = set(EVAL_FIELDS_IN_ORDER + SPECIAL_FIELDS + [BODY_SKILLS_FIELD])
    out: Dict[str, Any] = {}
    for k in keep_keys:
        if k in llm_obj:
            out[k] = llm_obj[k]
    return out


def _flatten_llm_for_csv(record_id: str, record_url: str, llm_filtered: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ONE row matching the fixed CSV schema.
    Missing fields are filled with "" (so Pipeline 2 always works).
    """
    out: Dict[str, Any] = {c: "" for c in _expected_csv_columns()}
    out["ID"] = record_id
    out["URL"] = record_url

    # Normal evaluated fields (each is "<Status> | <reason>")
    for f in EVAL_FIELDS_IN_ORDER:
        raw = llm_filtered.get(f, "")
        out[f] = raw
        status, reason = _split_status_reason(raw)
        out[f"{f}__status"] = status
        out[f"{f}__reason"] = reason

    # Special fields (also split)
    for f in SPECIAL_FIELDS:
        raw = llm_filtered.get(f, "")
        out[f] = raw
        status, reason = _split_status_reason(raw)
        out[f"{f}__status"] = status
        out[f"{f}__reason"] = reason

    # body_skills as JSON string list
    skills = _normalize_body_skills(llm_filtered.get(BODY_SKILLS_FIELD, None))
    out[BODY_SKILLS_FIELD] = json.dumps(skills, ensure_ascii=False)

    return out


# -----------------------------
# Main pipeline
# -----------------------------
def run(
    *,
    parameters_path: str = "configs/parameters.yaml",
    credentials_path: str = "configs/credentials.yaml",
    prompt_key: str = "job_posting_dq_eval_v1",
    max_rows: Optional[str | int] = None,
    force: bool = False,
    # preprocessing IO
    raw_postings_csv: str = "raw_data/Thailand_global_postings.csv",
    raw_jds_csv: str = "raw_data/Thailand_global_raw.csv",
    raw_skills_csv: str = "raw_data/Thailand_global_skills.csv",
    processed_postings_psv: str = "process_data/Thailand_global_postings.psv",
    processed_raw_psv: str = "process_data/Thailand_global_raw.psv",
    processed_skills_psv: str = "process_data/Thailand_global_skills.psv",
    # outputs
    output_jsonl_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    # record shaping
    exclude_cols_for_llm: Optional[List[str]] = None,
) -> None:
    params = load_parameters(parameters_path)
    creds = load_credentials(credentials_path)
    ensure_dirs(params)

    silence_client_logs = bool(getattr(getattr(params, "llm", None), "silence_client_lv_logs", False))
    configure_logging(level="INFO", silence_client_lv_logs=silence_client_logs)

    logger = get_logger(__name__)
    logger.info("Logging configured | silence_client_lv_logs=%s", str(silence_client_logs))

    cache_dir = getattr(getattr(params, "outputs", None), "cache_dir", None) or "artifacts/cache"

    output_jsonl_path = (
        output_jsonl_path
        or getattr(getattr(params, "outputs", None), "job_postings_dq_eval_jsonl", None)
        or "artifacts/reports/job_postings_dq_eval.jsonl"
    )
    legacy_output_csv = getattr(params, "output", None)
    output_csv_path = (
        output_csv_path
        or getattr(getattr(params, "outputs", None), "job_postings_dq_eval_csv", None)
        or legacy_output_csv
        or "artifacts/reports/job_postings_dq_eval.csv"
    )

    # Build merged dataframe (inputs)
    post_w_jd_df = _build_post_w_jd_df(
        raw_postings_csv=raw_postings_csv,
        raw_jds_csv=raw_jds_csv,
        raw_skills_csv=raw_skills_csv,
        processed_postings_psv=processed_postings_psv,
        processed_raw_psv=processed_raw_psv,
        processed_skills_psv=processed_skills_psv,
        logger=logger,
    )

    limit = _resolve_row_limit(params, max_rows)
    df_run = post_w_jd_df.head(limit).copy() if limit is not None else post_w_jd_df

    progress_every = _get_progress_every(params)
    max_workers = _get_max_workers(params)

    logger.info(
        "DQ Eval: rows=%d | limit=%s | progress_log_every=%d | cache_dir=%s | force=%s | max_workers=%d | prompt_key=%s",
        len(df_run),
        ("all" if limit is None else str(limit)),
        progress_every,
        cache_dir,
        str(force),
        max_workers,
        prompt_key,
    )

    # Model selection
    model_name = getattr(getattr(params, "llm", None), "model_name", None)
    if model_name and not os.environ.get("GEMINI_MODEL"):
        os.environ["GEMINI_MODEL"] = str(model_name)

    gemini_cfg = getattr(creds, "gemini", creds)

    # Thread-local Gemini client
    _tls = threading.local()

    def _get_thread_client_ctx():
        ctx = getattr(_tls, "client_ctx", None)
        if ctx is None:
            _tls.client_ctx = build_gemini_client(gemini_cfg)
        return _tls.client_ctx

    exclude_cols_for_llm = exclude_cols_for_llm or ["ID", "URL"]

    def _make_llm_record(row: pd.Series) -> Dict[str, Any]:
        d = {c: _as_str(row.get(c, "")) for c in df_run.columns.tolist()}
        for c in exclude_cols_for_llm:
            d.pop(c, None)
        return d

    def _process_one_row(i: int, row: pd.Series) -> Dict[str, Any]:
        record_id = _as_str(row.get("ID", "")) or str(i)
        record_url = _as_str(row.get("URL", ""))

        llm_record = _make_llm_record(row)
        record_json_str = json.dumps(llm_record, ensure_ascii=False)

        parsed = run_prompt_json(
            prompt_key=prompt_key,
            variables={"PASTE_RECORD_HERE": record_json_str},
            schema_model=AnyJsonObject,
            client_ctx=_get_thread_client_ctx(),
            temperature=float(getattr(params.llm, "temperature", 0.0)),
            max_retries=int(getattr(params.llm, "max_retries", 3)),
            json_only=bool(getattr(params.llm, "json_only", True)),
            cache_dir=cache_dir,
            cache_id=_cache_id(str(record_id), prompt_key=prompt_key),
            force=force,
        )

        llm_obj = parsed.model_dump()
        llm_filtered = _filter_llm_output_to_schema(llm_obj)

        # JSONL record keeps full filtered output + ID/URL
        jsonl_out = dict(llm_filtered)
        jsonl_out["ID"] = str(record_id)
        if record_url:
            jsonl_out["URL"] = record_url

        # CSV row is fully flattened with fixed schema
        csv_row = _flatten_llm_for_csv(str(record_id), record_url, llm_filtered)

        return {"jsonl": jsonl_out, "csv": csv_row}

    total = len(df_run)
    out_jsonl: List[Dict[str, Any]] = []
    out_csv_rows: List[Dict[str, Any]] = []

    if max_workers <= 1 or total <= 1:
        for i, (_, row) in enumerate(df_run.iterrows(), start=1):
            r = _process_one_row(i, row)
            out_jsonl.append(r["jsonl"])
            out_csv_rows.append(r["csv"])

            if (i == 1) or (progress_every > 0 and (i % progress_every == 0)) or (i == total):
                logger.info("DQ Eval: progress %d/%d (%0.1f%%)", i, total, (i / total * 100.0 if total else 100.0))
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, (_, row) in enumerate(df_run.iterrows(), start=1):
                fut = executor.submit(_process_one_row, i, row)
                futures[fut] = i

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    r = fut.result()
                    out_jsonl.append(r["jsonl"])
                    out_csv_rows.append(r["csv"])
                except Exception as e:
                    logger.exception("DQ Eval: worker failed | row_index=%s | error=%s", str(i), repr(e))
                    raise

                completed += 1
                if (completed == 1) or (progress_every > 0 and (completed % progress_every == 0)) or (completed == total):
                    logger.info(
                        "DQ Eval: progress %d/%d (%0.1f%%)",
                        completed,
                        total,
                        (completed / total * 100.0 if total else 100.0),
                    )

    # Deterministic ordering by ID
    out_jsonl.sort(key=lambda r: (r.get("ID", ""),))
    out_csv_rows.sort(key=lambda r: (r.get("ID", ""),))

    from functions.io.writers import write_jsonl

    # 1) JSONL
    logger.info("DQ Eval: writing JSONL=%d -> %s", len(out_jsonl), output_jsonl_path)
    write_jsonl(output_jsonl_path, out_jsonl)

    # 2) CSV with fixed schema
    logger.info("DQ Eval: writing CSV rows=%d -> %s", len(out_csv_rows), output_csv_path)
    df_report = pd.DataFrame(out_csv_rows)

    # Force exact column order (and ensure no extra cols)
    expected_cols = _expected_csv_columns()
    for c in expected_cols:
        if c not in df_report.columns:
            df_report[c] = ""
    df_report = df_report[expected_cols]

    df_report.to_csv(output_csv_path, index=False, encoding="utf-8")

    logger.info("DQ Eval: done")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pipeline 1: Job Posting DQ evaluation using Gemini (+CSV report).")
    parser.add_argument("--parameters-path", default="configs/parameters.yaml")
    parser.add_argument("--credentials-path", default="configs/credentials.yaml")
    parser.add_argument("--prompt-key", default="job_posting_dq_eval_v1")
    parser.add_argument("--max-rows", default=None, help='Integer or "all".')
    parser.add_argument("--force", action="store_true")

    # Optional overrides for IO
    parser.add_argument("--raw-postings-csv", default="raw_data/Thailand_global_postings.csv")
    parser.add_argument("--raw-jds-csv", default="raw_data/Thailand_global_raw.csv")
    parser.add_argument("--raw-skills-csv", default="raw_data/Thailand_global_skills.csv")
    parser.add_argument("--processed-postings-psv", default="process_data/Thailand_global_postings.psv")
    parser.add_argument("--processed-raw-psv", default="process_data/Thailand_global_raw.psv")
    parser.add_argument("--processed-skills-psv", default="process_data/Thailand_global_skills.psv")
    parser.add_argument("--output-jsonl-path", default=None)
    parser.add_argument("--output-csv-path", default=None)

    if argv is None:
        args, _unknown = parser.parse_known_args()
    else:
        args = parser.parse_args(argv)

    run(
        parameters_path=args.parameters_path,
        credentials_path=args.credentials_path,
        prompt_key=args.prompt_key,
        max_rows=args.max_rows,
        force=bool(args.force),
        raw_postings_csv=args.raw_postings_csv,
        raw_jds_csv=args.raw_jds_csv,
        raw_skills_csv=args.raw_skills_csv,
        processed_postings_psv=args.processed_postings_psv,
        processed_raw_psv=args.processed_raw_psv,
        processed_skills_psv=args.processed_skills_psv,
        output_jsonl_path=args.output_jsonl_path,
        output_csv_path=args.output_csv_path,
    )


if __name__ == "__main__":
    main()


__all__ = ["run", "main"]
