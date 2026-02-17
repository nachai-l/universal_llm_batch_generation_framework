# functions/core/export_outputs.py
"""
Export Outputs — core logic for Pipeline 5 (JSONL + PSV)

Intent
- Provide a small, testable core module that Pipeline 5 can call.
- Keep Pipeline 5 orchestration-only (paths + config + IO).

Scope
- Load/normalize indices from pipeline2_input.json
- Load group contexts from pipeline3_group_contexts.json (optional)
- Read pipeline4_manifest.json + llm_outputs/*.json (success files)
- Build export records (row-output + group-output expansion)
- Sort deterministically
- Flatten for PSV with stable column ordering
- Drop heavy columns when requested (thin outputs)

Non-goals
- No file IO here (handled by pipeline_5_export_outputs.py).
- No pandas dependency here (pipeline can use pandas to write PSV; we only produce dict rows).

Determinism
- Stable selection + ordering rules
- Deterministic JSON stringification for dict/list fields
- PSV-safe sanitization (no embedded newlines/tabs in row values)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from functions.utils.text import json_stringify_if_needed, sanitize_psv_value


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

HEAVY_EXPORT_COLUMNS_DEFAULT: Tuple[str, ...] = (
    "group_rows_json",
    "group_context_id",
    "group_context",
    "group_context_meta_json",
    "questions_json",
)

GROUP_HELPER_COLS: Tuple[str, ...] = (
    "group_key",
    "grouping_column",
    "n_group_rows",
    "group_row_indices_json",
    "group_rows_json",
    "group_context_id",
    "group_context",
    "group_context_meta_json",
    "questions_json",
)

JUDGE_META_COLS: Tuple[str, ...] = (
    "judge_verdict",
    "judge_score",
    "judge_reasons_json",
    "meta_cache_id",
    "meta_work_id",
    "meta_group_key",
    "meta_row_index",
    "meta_group_context_id",
    "meta_model",
    "meta_temperature",
    "meta_created_at_utc",
)


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Pipeline2Index:
    input_columns: List[str]
    row_index_to_record: Dict[int, Dict[str, Any]]
    grouping_column_inferred: Optional[str]


@dataclass(frozen=True)
class GroupContextIndex:
    # group_context_id -> group_context object
    by_id: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class ExportDiagnostics:
    n_missing_output_file: int
    n_row_outputs_seen: int
    n_row_index_out_of_range: int
    n_group_outputs_seen: int
    n_group_context_missing: int
    n_group_context_row_indices_missing: int
    n_expanded_rows: int
    n_selected_question_missing: int


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def infer_grouping_column(p2_meta: Any, records: List[Any]) -> Optional[str]:
    """
    Best-effort inference for grouping column used in group joins.
    Tries:
      - meta.grouping.column
      - meta.grouping_column
      - fallback: 'Role Track Example Name' if present in input records
    """
    if isinstance(p2_meta, dict):
        g = p2_meta.get("grouping")
        if isinstance(g, dict):
            col = g.get("column")
            if isinstance(col, str) and col.strip():
                return col.strip()

        col2 = p2_meta.get("grouping_column")
        if isinstance(col2, str) and col2.strip():
            return col2.strip()

    fallback = "Role Track Example Name"
    for r in records[:50]:
        if isinstance(r, dict) and fallback in r:
            return fallback

    return None


# ---------------------------------------------------------------------
# Load indices
# ---------------------------------------------------------------------


def build_pipeline2_index(p2_obj: Dict[str, Any]) -> Pipeline2Index:
    """
    Build:
    - input_columns
    - row_index_to_record mapping (record position is row_index)
    - inferred grouping column
    """
    meta = p2_obj.get("meta") if isinstance(p2_obj, dict) else None
    records = p2_obj.get("records") if isinstance(p2_obj, dict) else None
    if records is None:
        records = p2_obj.get("rows") if isinstance(p2_obj, dict) else None
    if not isinstance(records, list):
        raise ValueError("pipeline2_input.json malformed: records is not a list")

    cols: List[str] = []
    if isinstance(meta, dict) and isinstance(meta.get("columns"), list):
        cols = [str(c) for c in meta.get("columns") if c is not None]
    if not cols and records and isinstance(records[0], dict):
        cols = [str(c) for c in records[0].keys()]

    idx: Dict[int, Dict[str, Any]] = {}
    for i, rec in enumerate(records):
        if isinstance(rec, dict):
            idx[int(i)] = dict(rec)

    grouping_col = infer_grouping_column(meta, records)

    return Pipeline2Index(input_columns=cols, row_index_to_record=idx, grouping_column_inferred=grouping_col)


def build_group_context_index(p3_group_contexts_obj: Any) -> GroupContextIndex:
    """
    Supports either:
      - list[...] (latest)
      - {"groups": [...]} (older)
    """
    groups: List[Any] = []
    if isinstance(p3_group_contexts_obj, list):
        groups = p3_group_contexts_obj
    elif isinstance(p3_group_contexts_obj, dict) and isinstance(p3_group_contexts_obj.get("groups"), list):
        groups = p3_group_contexts_obj["groups"]  # type: ignore[assignment]

    out: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        if not isinstance(g, dict):
            continue
        gcid = str(g.get("group_context_id") or "").strip()
        if not gcid:
            continue
        out[gcid] = dict(g)

    return GroupContextIndex(by_id=out)


# ---------------------------------------------------------------------
# Selection logic (group-output expansion)
# ---------------------------------------------------------------------


def select_question_for_row(parsed_group: Dict[str, Any], input_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort selection of the question object for a row.

    Strategy:
    - parsed_group.questions is list[dict]
    - match question_id == input_row["Question ID"] (string compare)
    """
    qid = str(input_row.get("Question ID") or "").strip()
    qs = parsed_group.get("questions")

    if not qid or not isinstance(qs, list):
        return {}

    for q in qs:
        if not isinstance(q, dict):
            continue
        if str(q.get("question_id") or "").strip() == qid:
            return dict(q)

    return {}


# ---------------------------------------------------------------------
# Normalization (rich record)
# ---------------------------------------------------------------------


def extract_export_record(
    out_obj: Dict[str, Any],
    *,
    input_row: Optional[Dict[str, Any]],
    input_group_rows: Optional[List[Dict[str, Any]]] = None,
    input_group_row_indices: Optional[List[int]] = None,
    grouping_column: Optional[str] = None,
    group_context: Optional[str] = None,
    group_context_meta: Optional[Dict[str, Any]] = None,
    parsed_group: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize one llm_outputs/*.json into a consistent rich export record.

    For expanded group-output rows:
    - input is set
    - parsed is the per-row selected question dict (best-effort)
    - parsed_group is the full group parsed payload (kept for traceability)
    """
    meta = out_obj.get("meta") if isinstance(out_obj, dict) else None
    judge = out_obj.get("judge") if isinstance(out_obj, dict) else None

    if not isinstance(meta, dict):
        meta = {}
    if judge is not None and not isinstance(judge, dict):
        judge = None

    parsed_row: Dict[str, Any] = {}
    if isinstance(parsed_group, dict):
        if isinstance(parsed_group.get("questions"), list) and isinstance(input_row, dict):
            parsed_row = select_question_for_row(parsed_group, input_row)
        else:
            parsed_row = dict(parsed_group)

    return {
        "input": dict(input_row) if isinstance(input_row, dict) else None,
        "input_group_rows": list(input_group_rows) if isinstance(input_group_rows, list) else None,
        "input_group_row_indices": list(input_group_row_indices) if isinstance(input_group_row_indices, list) else None,
        "grouping_column": str(grouping_column) if isinstance(grouping_column, str) and grouping_column else None,
        "group_context": str(group_context) if isinstance(group_context, str) else None,
        "group_context_meta": dict(group_context_meta) if isinstance(group_context_meta, dict) else None,
        "parsed": dict(parsed_row),
        "parsed_group": dict(parsed_group) if isinstance(parsed_group, dict) else None,
        "judge": dict(judge) if isinstance(judge, dict) else None,
        "meta": dict(meta),
    }


def sort_key_for_export(rec: Dict[str, Any]) -> Tuple[int, str, str]:
    meta = rec.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    row_index = safe_int(meta.get("row_index"))
    row_index_sort = 10**12 if row_index is None else int(row_index)

    group_key = str(meta.get("group_key") or "")
    work_id = str(meta.get("work_id") or "")
    return row_index_sort, group_key, work_id


# ---------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------


def build_export_records(
    *,
    pipeline2: Pipeline2Index,
    group_contexts: GroupContextIndex,
    success_files: Sequence[str],
    outputs_dir: Path,
    read_json_func,
) -> Tuple[List[Dict[str, Any]], ExportDiagnostics]:
    """
    Build sorted rich export records.

    read_json_func is injected for testability (pipeline will pass functions.io.readers.read_json).
    """
    export_records: List[Dict[str, Any]] = []

    n_missing_output_file = 0
    n_group_outputs_seen = 0
    n_group_context_missing = 0
    n_group_context_row_indices_missing = 0
    n_expanded_rows = 0
    n_selected_question_missing = 0

    n_row_outputs_seen = 0
    n_row_index_out_of_range = 0

    for fname in success_files:
        if not isinstance(fname, str) or not fname.endswith(".json"):
            continue

        fpath = outputs_dir / fname
        if not fpath.exists():
            n_missing_output_file += 1
            continue

        obj = read_json_func(fpath)
        if not isinstance(obj, dict):
            continue

        meta = obj.get("meta")
        if not isinstance(meta, dict):
            meta = {}

        row_index = safe_int(meta.get("row_index"))
        group_key = str(meta.get("group_key") or "").strip()
        group_context_id = str(meta.get("group_context_id") or "").strip()

        parsed_full = obj.get("parsed") if isinstance(obj.get("parsed"), dict) else {}

        if row_index is not None:
            n_row_outputs_seen += 1
            input_row = pipeline2.row_index_to_record.get(int(row_index))
            if input_row is None:
                n_row_index_out_of_range += 1

            export_records.append(
                extract_export_record(
                    obj,
                    input_row=input_row,
                    grouping_column=pipeline2.grouping_column_inferred,
                    group_context=None,
                    group_context_meta=None,
                    parsed_group=parsed_full,
                )
            )
            continue

        # Group-output expansion path
        n_group_outputs_seen += 1

        gc = group_contexts.by_id.get(group_context_id) if group_context_id else None
        if not isinstance(gc, dict):
            n_group_context_missing += 1
            export_records.append(
                extract_export_record(
                    obj,
                    input_row=None,
                    grouping_column=pipeline2.grouping_column_inferred,
                    group_context=None,
                    group_context_meta=None,
                    parsed_group=parsed_full,
                )
            )
            continue

        gc_meta = gc.get("meta") if isinstance(gc.get("meta"), dict) else {}
        gc_context = gc.get("context") if isinstance(gc.get("context"), str) else ""

        gc_grouping_column = gc_meta.get("grouping_column")
        grouping_column = (
            str(gc_grouping_column).strip()
            if isinstance(gc_grouping_column, str) and gc_grouping_column.strip()
            else pipeline2.grouping_column_inferred
        )

        row_indices_used = gc_meta.get("row_indices_used")
        if not isinstance(row_indices_used, list) or not row_indices_used:
            row_indices_used = gc_meta.get("row_indices_all")
        if not isinstance(row_indices_used, list) or not row_indices_used:
            n_group_context_row_indices_missing += 1
            row_indices_used = []

        grp_rows: List[Dict[str, Any]] = []
        grp_idx: List[int] = []
        for ri in row_indices_used:
            rii = safe_int(ri)
            if rii is None:
                continue
            rrow = pipeline2.row_index_to_record.get(int(rii))
            if isinstance(rrow, dict):
                grp_rows.append(dict(rrow))
                grp_idx.append(int(rii))

        for rii in grp_idx:
            input_row = pipeline2.row_index_to_record.get(int(rii))
            if not isinstance(input_row, dict):
                continue

            row_meta = dict(meta)
            row_meta["row_index"] = int(rii)
            row_meta["group_key"] = group_key or str(gc.get("group_key") or "").strip()
            row_meta["group_context_id"] = group_context_id or str(gc.get("group_context_id") or "").strip()

            row_obj = dict(obj)
            row_obj["meta"] = row_meta

            rec = extract_export_record(
                row_obj,
                input_row=input_row,
                input_group_rows=grp_rows,
                input_group_row_indices=grp_idx,
                grouping_column=grouping_column,
                group_context=gc_context,
                group_context_meta=gc_meta if isinstance(gc_meta, dict) else None,
                parsed_group=parsed_full,
            )

            if isinstance(parsed_full, dict) and isinstance(parsed_full.get("questions"), list):
                if not rec.get("parsed"):
                    n_selected_question_missing += 1

            export_records.append(rec)
            n_expanded_rows += 1

    export_records_sorted = sorted(export_records, key=sort_key_for_export)

    diag = ExportDiagnostics(
        n_missing_output_file=int(n_missing_output_file),
        n_row_outputs_seen=int(n_row_outputs_seen),
        n_row_index_out_of_range=int(n_row_index_out_of_range),
        n_group_outputs_seen=int(n_group_outputs_seen),
        n_group_context_missing=int(n_group_context_missing),
        n_group_context_row_indices_missing=int(n_group_context_row_indices_missing),
        n_expanded_rows=int(n_expanded_rows),
        n_selected_question_missing=int(n_selected_question_missing),
    )
    return export_records_sorted, diag


# ---------------------------------------------------------------------
# Flattening for PSV
# ---------------------------------------------------------------------

def flatten_for_psv(
    rec: Dict[str, Any],
    *,
    input_columns: Sequence[str],
    drop_columns: Optional[Sequence[str]] = None,
    include_group_columns: bool = False,
    include_questions_json: bool = False,
    include_group_trace_ids: bool = False,
    parsed_prefix_on_collision: str = "parsed_",
) -> Dict[str, Any]:
    """
    Produce a flat row for PSV with stable semantics.

    Collision policy (FIX for 'Question ID' vs 'question_id'):
    - If a parsed key would collide *semantically* with an input column header
      (e.g., "Question ID" vs "question_id"), we rename the parsed field to:
        {parsed_prefix_on_collision}{parsed_key}
      Default prefix: "parsed_"

    This preserves BOTH:
      - Input "Question ID"
      - Parsed "parsed_question_id"
    """
    import re  # local import to keep function copy/paste-ready

    def _norm_header(s: Any) -> str:
        # Normalize headers to detect semantic collisions:
        # "Question ID" == "question_id" == "Question-ID"
        t = str(s or "").strip().lower()
        t = re.sub(r"[^a-z0-9]+", "_", t)
        t = re.sub(r"_+", "_", t).strip("_")
        return t

    out: Dict[str, Any] = {}

    # -----------------------------
    # Input columns (stable order)
    # -----------------------------
    input_row = rec.get("input")
    if isinstance(input_row, dict):
        for c in input_columns:
            out[str(c)] = input_row.get(c, "")
    else:
        for c in input_columns:
            out[str(c)] = ""

    input_norm_map = {_norm_header(c): str(c) for c in input_columns}

    # -----------------------------
    # Optional group trace/helpers
    # -----------------------------
    meta = rec.get("meta")
    meta = meta if isinstance(meta, dict) else {}

    if include_group_trace_ids:
        out["group_key"] = str(meta.get("group_key") or "")
        out["grouping_column"] = str(rec.get("grouping_column") or "")
        out["n_group_rows"] = len(rec.get("input_group_rows")) if isinstance(rec.get("input_group_rows"), list) else 0
        out["group_row_indices_json"] = json_stringify_if_needed(
            rec.get("input_group_row_indices") if isinstance(rec.get("input_group_row_indices"), list) else []
        )

    if include_group_columns:
        grp_rows = rec.get("input_group_rows")
        grp_idx = rec.get("input_group_row_indices")

        out["group_rows_json"] = json_stringify_if_needed(grp_rows if isinstance(grp_rows, list) else [])
        out["group_context_id"] = str(meta.get("group_context_id") or "")
        out["group_context"] = str(rec.get("group_context") or "")
        out["group_context_meta_json"] = json_stringify_if_needed(rec.get("group_context_meta") or {})

    if include_questions_json:
        parsed_group = rec.get("parsed_group")
        if isinstance(parsed_group, dict) and "questions" in parsed_group:
            out["questions_json"] = json_stringify_if_needed(parsed_group.get("questions"))
        else:
            out["questions_json"] = json_stringify_if_needed([])

    # -----------------------------
    # Parsed per-row fields
    # -----------------------------
    parsed = rec.get("parsed")
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            key = str(k)

            # Exact collision (already present)
            if key in out:
                key = f"{parsed_prefix_on_collision}{key}"

            # Semantic collision: "Question ID" vs "question_id"
            norm_key = _norm_header(key)
            if norm_key in input_norm_map:
                # Only rename if it isn't already the *same* header as input.
                # (If input column itself is exactly "question_id", then this will still rename,
                # because we want to preserve the input column as canonical.)
                key = f"{parsed_prefix_on_collision}{key}"

            out[key] = json_stringify_if_needed(v)

    # -----------------------------
    # Judge fields
    # -----------------------------
    judge = rec.get("judge")
    if isinstance(judge, dict):
        out["judge_verdict"] = judge.get("verdict", "")
        out["judge_score"] = judge.get("score", "")
        out["judge_reasons_json"] = json_stringify_if_needed(judge.get("reasons", []))
    else:
        out["judge_verdict"] = ""
        out["judge_score"] = ""
        out["judge_reasons_json"] = json_stringify_if_needed([])

    # -----------------------------
    # Meta fields (always)
    # -----------------------------
    out["meta_cache_id"] = meta.get("cache_id", "")
    out["meta_work_id"] = meta.get("work_id", "")
    out["meta_group_key"] = meta.get("group_key", "")
    out["meta_row_index"] = meta.get("row_index", "")
    out["meta_group_context_id"] = meta.get("group_context_id", "")
    out["meta_model"] = meta.get("model", "")
    out["meta_temperature"] = meta.get("temperature", "")
    out["meta_created_at_utc"] = meta.get("created_at_utc", "")

    # -----------------------------
    # Drop requested columns (final override)
    # -----------------------------
    if drop_columns:
        for c in drop_columns:
            out.pop(str(c), None)

    # -----------------------------
    # PSV safety (1 row = 1 line)
    # -----------------------------
    out = {k: sanitize_psv_value(v) for k, v in out.items()}
    return out


def compute_psv_column_order(
    *,
    input_columns: Sequence[str],
    flat_rows: Sequence[Mapping[str, Any]],
    drop_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Deterministic PSV columns:
      - input columns in source order
      - group helper columns (if present & not dropped)
      - parsed columns (alphabetical)
      - judge/meta columns in fixed order
    """
    if not flat_rows:
        mid_cols = [c for c in GROUP_HELPER_COLS if c not in set(drop_columns or ())]
        return list(input_columns) + mid_cols + list(JUDGE_META_COLS)

    current_cols = set()
    for r in flat_rows:
        current_cols.update(str(k) for k in r.keys())

    drops = set(str(x) for x in (drop_columns or ()))

    desired_prefix = [
        c for c in input_columns
        if str(c) in current_cols and str(c) not in drops
    ]

    mid_cols = [
        c for c in GROUP_HELPER_COLS
        if c in current_cols and c not in drops
    ]

    tail_cols = [
        c for c in JUDGE_META_COLS
        if c in current_cols and c not in drops
    ]

    blocked = set(desired_prefix) | set(mid_cols) | set(tail_cols)

    parsed_cols = sorted([
        c for c in current_cols
        if c not in blocked and c not in drops
    ])

    return desired_prefix + mid_cols + parsed_cols + tail_cols


__all__ = [
    "Pipeline2Index",
    "GroupContextIndex",
    "ExportDiagnostics",
    "HEAVY_EXPORT_COLUMNS_DEFAULT",
    "build_pipeline2_index",
    "build_group_context_index",
    "build_export_records",
    "flatten_for_psv",
    "compute_psv_column_order",
]
