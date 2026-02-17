# functions/batch/pipeline_6_write_report.py
"""
Pipeline 6 — Write Report (MD + optional HTML)

Intent
- Read exported outputs (Pipeline 5 JSONL)
- Summarize run + judge results + coverage
- Write deterministic artifacts:
  - artifacts/reports/report.md
  - artifacts/reports/report.html (optional)
  - artifacts/cache/pipeline6_manifest.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from functions.core.reporting import compute_report_stats, render_report_md
from functions.io.html_rendering import render_report_html
from functions.io.readers import read_json
from functions.io.writers import ensure_parent_dir, write_json
from functions.utils.config import load_parameters
from functions.utils.logging import configure_logging_from_params, get_logger
from functions.utils.paths import repo_root_from_parameters_path, resolve_path


PIPELINE5_MANIFEST_DEFAULT = "artifacts/cache/pipeline5_manifest.json"
PIPELINE6_MANIFEST_DEFAULT = "artifacts/cache/pipeline6_manifest.json"


# -----------------------------
# Helpers
# -----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_jsonl_records(path: str | Path) -> List[Dict[str, Any]]:
    """
    Materialize JSONL records into memory once.
    Keeps behavior deterministic and avoids generator re-consumption issues.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {str(p)}")

    records: List[Dict[str, Any]] = []

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {str(p)} line={line_no}: {e}") from e

            if isinstance(obj, dict):
                records.append(obj)

    return records


# -----------------------------
# Main
# -----------------------------

def main(
    *,
    parameters_path: str | Path = "configs/parameters.yaml",
    pipeline5_manifest_path: str | Path = PIPELINE5_MANIFEST_DEFAULT,
    out_manifest_path: str | Path = PIPELINE6_MANIFEST_DEFAULT,
) -> int:
    params = load_parameters(parameters_path)

    configure_logging_from_params(
        params,
        level=str(getattr(getattr(params, "run", None), "log_level", "INFO")),
        log_file=getattr(getattr(params, "run", None), "log_file", None),
    )
    logger = get_logger(__name__)

    # ---------------------------------
    # Early exit if report disabled
    # ---------------------------------
    rep = getattr(params, "report", None)
    if rep is None or not bool(getattr(rep, "enabled", True)):
        logger.info("Pipeline 6 skipped (report.enabled=false)")
        return 0

    repo_root = repo_root_from_parameters_path(parameters_path)

    # ---------------------------------
    # Load Pipeline 5 manifest
    # ---------------------------------
    p5_path = resolve_path(pipeline5_manifest_path, base_dir=repo_root)
    p5 = read_json(p5_path)
    meta5 = p5.get("meta") if isinstance(p5, dict) else {}
    if not isinstance(meta5, dict):
        meta5 = {}

    output_jsonl = meta5.get("outputs_jsonl")
    output_psv = meta5.get("outputs_psv")
    pipeline4_manifest_path = meta5.get("pipeline4_manifest_path")
    pipeline2_ingest_path = meta5.get("pipeline2_ingest_path")

    if not output_jsonl:
        raise RuntimeError("pipeline5_manifest missing meta.outputs_jsonl")

    # ---------------------------------
    # Load Pipeline 4 meta (judge hint)
    # ---------------------------------
    judge_enabled_hint = False
    pipeline4_meta: Dict[str, Any] = {}

    if pipeline4_manifest_path:
        p4 = read_json(Path(pipeline4_manifest_path))
        pipeline4_meta = (p4.get("meta") if isinstance(p4, dict) else {}) or {}
        if isinstance(pipeline4_meta, dict):
            judge_enabled_hint = bool(pipeline4_meta.get("judge_enabled", False))

    # ---------------------------------
    # Load Pipeline 2 meta (input info)
    # ---------------------------------
    pipeline2_meta: Dict[str, Any] = {}
    if pipeline2_ingest_path:
        p2 = read_json(Path(pipeline2_ingest_path))
        pipeline2_meta = (p2.get("meta") if isinstance(p2, dict) else {}) or {}

    # ---------------------------------
    # Report configuration
    # ---------------------------------
    sample_per_group = int(getattr(rep, "sample_per_group", 2))
    include_full_examples = bool(getattr(rep, "include_full_examples", False))
    write_html = bool(getattr(rep, "write_html", True))
    max_reason_examples = int(getattr(rep, "max_reason_examples", 5))

    # ---------------------------------
    # Load JSONL records (materialized)
    # ---------------------------------
    records = _load_jsonl_records(output_jsonl)

    # ---------------------------------
    # Compute report stats
    # ---------------------------------
    stats = compute_report_stats(
        records=records,
        sample_per_role=sample_per_group,
        include_full_examples=include_full_examples,
        judge_enabled_hint=judge_enabled_hint,
        max_reason_examples=max_reason_examples,
    )

    # ---------------------------------
    # Resolve report paths
    # ---------------------------------
    md_path = resolve_path(rep.md_path, base_dir=repo_root)
    html_path = resolve_path(rep.html_path, base_dir=repo_root)

    # ---------------------------------
    # Prepare report meta
    # ---------------------------------
    report_meta: Dict[str, Any] = {
        "pipeline": 6,
        "generated_at_utc": _utc_now_iso(),
        "parameters_path": str(Path(parameters_path)),
        "pipeline5_manifest_path": str(p5_path),
        "pipeline4_manifest_path": str(pipeline4_manifest_path) if pipeline4_manifest_path else None,
        "pipeline2_ingest_path": str(pipeline2_ingest_path) if pipeline2_ingest_path else None,
        "output_jsonl": str(output_jsonl),
        "output_psv": str(output_psv) if output_psv else None,
        "input_summary": {
            "n_rows": pipeline2_meta.get("n_rows") if isinstance(pipeline2_meta, dict) else None,
            "n_cols": pipeline2_meta.get("n_cols") if isinstance(pipeline2_meta, dict) else None,
            "columns": pipeline2_meta.get("columns") if isinstance(pipeline2_meta, dict) else None,
            "input_path": pipeline2_meta.get("input_path") if isinstance(pipeline2_meta, dict) else None,
            "input_format": pipeline2_meta.get("input_format") if isinstance(pipeline2_meta, dict) else None,
        },
        "pipeline4_summary": {
            "model_name": pipeline4_meta.get("model_name") if isinstance(pipeline4_meta, dict) else None,
            "temperature": pipeline4_meta.get("temperature") if isinstance(pipeline4_meta, dict) else None,
            "judge_enabled": judge_enabled_hint,
            "max_workers": pipeline4_meta.get("max_workers") if isinstance(pipeline4_meta, dict) else None,
        },
        "report_config": {
            "write_html": write_html,
            "sample_per_role": sample_per_group,
            "include_full_examples": include_full_examples,
            "max_reason_examples": max_reason_examples,
        },
    }

    # ---------------------------------
    # Render Markdown
    # ---------------------------------
    md_text = render_report_md(
        meta=report_meta,
        stats=stats,
        sample_per_group=sample_per_group,
        include_full_examples=include_full_examples,
        max_reason_examples=max_reason_examples,
    )

    ensure_parent_dir(md_path)
    Path(md_path).write_text(md_text, encoding="utf-8")
    logger.info("Wrote report markdown: %s", str(md_path))

    # ---------------------------------
    # Optional HTML
    # ---------------------------------
    html_written: Optional[str] = None

    if write_html:
        html_text = render_report_html(md_text=md_text, title="Pipeline 6 Report")
        ensure_parent_dir(html_path)
        Path(html_path).write_text(html_text, encoding="utf-8")
        html_written = str(html_path)
        logger.info("Wrote report html: %s", str(html_path))

    # ---------------------------------
    # Manifest (deterministic ordering)
    # ---------------------------------
    manifest: Dict[str, Any] = {
        "meta": {
            "pipeline": 6,
            "generated_at_utc": report_meta["generated_at_utc"],
            "parameters_path": report_meta["parameters_path"],
            "pipeline5_manifest_path": report_meta["pipeline5_manifest_path"],
            "output_jsonl": report_meta["output_jsonl"],
            "output_psv": report_meta["output_psv"],
            "report_md": str(md_path),
            "report_html": html_written,
        },
        "counts": {
            "n_records_jsonl": int(stats.n_records),
            "judge_enabled": bool(stats.judge.enabled),
            "n_pass": int(stats.judge.n_pass),
            "n_fail": int(stats.judge.n_fail),
        },
        "group_counts": {
            "by_role": stats.groups.by_role,
            "by_set": stats.groups.by_set,
            "by_type": stats.groups.by_type,
        },
        "data_quality": stats.data_quality,
    }

    out_manifest = resolve_path(out_manifest_path, base_dir=repo_root)
    write_json(out_manifest, manifest)

    logger.info(
        "Pipeline 6 completed | report_md=%s report_html=%s manifest=%s",
        str(md_path),
        str(html_written),
        str(out_manifest),
    )

    return 0


__all__ = ["main"]
