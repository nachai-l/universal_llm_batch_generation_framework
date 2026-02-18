# functions/batch/pipeline_5_export_outputs.py
"""
Pipeline 5 — Export Outputs (JSONL + PSV)

Thin orchestration layer.

Core logic moved to:
    functions/core/export_outputs.py

Responsibilities here:
- Load config + resolve paths
- Load pipeline2 / pipeline3 / pipeline4 artifacts
- Call core builder
- Write JSONL + PSV
- Write pipeline5 manifest
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from functions.core.export_outputs import (
    HEAVY_EXPORT_COLUMNS_DEFAULT,
    build_export_records,
    build_group_context_index,
    build_group_context_index_from_work_items,
    build_pipeline2_index,
    compute_psv_column_order,
    flatten_for_psv,
)
from functions.io.readers import read_json
from functions.io.writers import write_json, write_jsonl, write_psv
from functions.utils.config import ensure_dirs, load_parameters
from functions.utils.logging import get_logger
from functions.utils.paths import repo_root_from_parameters_path, resolve_path


PIPELINE2_INGEST_PATH_DEFAULT = "artifacts/cache/pipeline2_input.json"
PIPELINE3_GROUP_CONTEXTS_PATH_DEFAULT = "artifacts/cache/pipeline3_group_contexts.json"
PIPELINE4_MANIFEST_PATH_DEFAULT = "artifacts/cache/pipeline4_manifest.json"
PIPELINE4_OUTPUTS_DIR_DEFAULT = "artifacts/cache/llm_outputs"
PIPELINE5_MANIFEST_PATH_DEFAULT = "artifacts/cache/pipeline5_manifest.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(
    *,
    parameters_path: str | Path = "configs/parameters.yaml",
    pipeline2_ingest_path: str | Path = PIPELINE2_INGEST_PATH_DEFAULT,
    pipeline3_group_contexts_path: str | Path = PIPELINE3_GROUP_CONTEXTS_PATH_DEFAULT,
    pipeline4_manifest_path: str | Path = PIPELINE4_MANIFEST_PATH_DEFAULT,
    pipeline4_outputs_dir: str | Path = PIPELINE4_OUTPUTS_DIR_DEFAULT,
    pipeline5_manifest_path: str | Path = PIPELINE5_MANIFEST_PATH_DEFAULT,
) -> int:
    logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    params = load_parameters(parameters_path)
    ensure_dirs(params)

    repo_root = repo_root_from_parameters_path(parameters_path)

    p2_path = Path(resolve_path(pipeline2_ingest_path, base_dir=repo_root))
    p3_gc_path = Path(resolve_path(pipeline3_group_contexts_path, base_dir=repo_root))
    p4m_path = Path(resolve_path(pipeline4_manifest_path, base_dir=repo_root))
    p4_out_dir = Path(resolve_path(pipeline4_outputs_dir, base_dir=repo_root))

    out_jsonl = Path(resolve_path(params.outputs.jsonl_path, base_dir=repo_root))
    out_psv = Path(resolve_path(params.outputs.psv_path, base_dir=repo_root))
    p5_manifest = Path(resolve_path(pipeline5_manifest_path, base_dir=repo_root))

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    p2_obj = read_json(p2_path)
    p2_index = build_pipeline2_index(p2_obj)

    p3_gc_obj = read_json(p3_gc_path) if p3_gc_path.exists() else []
    group_index = build_group_context_index(p3_gc_obj)

    # Fallback: if group contexts are empty, extract from work items
    if not group_index.by_id:
        p3_work_items_path = Path(resolve_path("artifacts/cache/pipeline3_work_items.json", base_dir=repo_root))
        if p3_work_items_path.exists():
            work_items_obj = read_json(p3_work_items_path)
            if isinstance(work_items_obj, dict) and isinstance(work_items_obj.get("items"), list):
                group_index = build_group_context_index_from_work_items(work_items_obj["items"])

    p4_manifest = read_json(p4m_path)
    outputs = p4_manifest.get("outputs") if isinstance(p4_manifest, dict) else None
    if not isinstance(outputs, dict):
        raise ValueError("pipeline4_manifest.json malformed: outputs missing or invalid")

    success_files = outputs.get("success_files")
    if not isinstance(success_files, list):
        raise ValueError("pipeline4_manifest.json malformed: success_files missing")

    # ------------------------------------------------------------------
    # Build export records (core)
    # ------------------------------------------------------------------
    export_records, diag = build_export_records(
        pipeline2=p2_index,
        group_contexts=group_index,
        success_files=success_files,
        outputs_dir=p4_out_dir,
        read_json_func=read_json,
    )

    # ------------------------------------------------------------------
    # Write JSONL (rich records)
    # ------------------------------------------------------------------
    write_jsonl(out_jsonl, export_records)

    # ------------------------------------------------------------------
    # Build PSV rows (thin by default)
    # ------------------------------------------------------------------
    drop_heavy = getattr(params.outputs, "drop_heavy_columns", True)

    drop_columns: List[str] = []
    if drop_heavy:
        drop_columns = list(HEAVY_EXPORT_COLUMNS_DEFAULT)

    flat_rows = [
        flatten_for_psv(
            r,
            input_columns=p2_index.input_columns,
            drop_columns=drop_columns,
        )
        for r in export_records
    ]

    df = pd.DataFrame.from_records(flat_rows)

    ordered_cols = compute_psv_column_order(
        input_columns=p2_index.input_columns,
        flat_rows=flat_rows,
        drop_columns=drop_columns,
    )

    df = df.reindex(columns=ordered_cols)
    write_psv(out_psv, df)

    # ------------------------------------------------------------------
    # Determine detected mode
    # ------------------------------------------------------------------
    any_row_index = any(
        isinstance(r.get("meta"), dict)
        and r["meta"].get("row_index") is not None
        for r in export_records
    )
    detected_mode = "row_output_expanded" if any_row_index else "group_output_unexpanded"

    # ------------------------------------------------------------------
    # Write Pipeline 5 manifest
    # ------------------------------------------------------------------
    manifest = {
        "meta": {
            "pipeline": 5,
            "generated_at_utc": _utc_now_iso(),
            "parameters_path": str(Path(parameters_path)),
            "pipeline2_ingest_path": str(p2_path),
            "pipeline3_group_contexts_path": str(p3_gc_path),
            "pipeline4_manifest_path": str(p4m_path),
            "pipeline4_outputs_dir": str(p4_out_dir),
            "outputs_jsonl": str(out_jsonl),
            "outputs_psv": str(out_psv),
            "detected_mode": detected_mode,
            "grouping_column": p2_index.grouping_column_inferred,
            "drop_heavy_columns": bool(drop_heavy),
        },
        "counts": {
            "n_success_files": int(len([x for x in success_files if isinstance(x, str)])),
            "n_exported_records": int(len(export_records)),
            "n_missing_output_file": diag.n_missing_output_file,
            "n_row_outputs_seen": diag.n_row_outputs_seen,
            "n_row_index_out_of_range": diag.n_row_index_out_of_range,
            "n_group_outputs_seen": diag.n_group_outputs_seen,
            "n_group_context_missing": diag.n_group_context_missing,
            "n_group_context_row_indices_missing": diag.n_group_context_row_indices_missing,
            "n_expanded_rows": diag.n_expanded_rows,
            "n_selected_question_missing": diag.n_selected_question_missing,
        },
        "input": {
            "input_columns": list(p2_index.input_columns),
        },
    }

    write_json(p5_manifest, manifest)

    logger.info(
        "Pipeline 5 completed | mode=%s exported=%d jsonl=%s psv=%s drop_heavy=%s manifest=%s",
        detected_mode,
        int(len(export_records)),
        str(out_jsonl),
        str(out_psv),
        bool(drop_heavy),
        str(p5_manifest),
    )

    return 0


__all__ = ["main"]
