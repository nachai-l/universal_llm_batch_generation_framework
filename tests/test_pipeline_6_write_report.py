# tests/test_pipeline_6_write_report.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from functions.batch.pipeline_6_write_report import main as pipeline6_main


def _write(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _write_json(p: Path, obj: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(p: Path, records: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")


def _make_repo_skeleton(repo: Path) -> None:
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "cache").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "outputs").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "reports").mkdir(parents=True, exist_ok=True)


def _make_min_pipeline2_input(repo: Path, n_rows: int = 2) -> Path:
    p2 = {
        "meta": {
            "n_rows": n_rows,
            "n_cols": 5,
            "columns": ["Role Track Example Name", "Assumed Lv", "Question Set", "Question ID", "Question Type"],
            "input_path": "raw_data/input.csv",
            "input_format": "csv",
        },
        "records": [],
        "n_rows": n_rows,
        "n_cols": 5,
    }
    p2_path = repo / "artifacts" / "cache" / "pipeline2_input.json"
    _write_json(p2_path, p2)
    return p2_path


def _make_min_pipeline4_manifest(repo: Path, *, judge_enabled: bool = True) -> Path:
    p4 = {
        "meta": {"judge_enabled": judge_enabled, "model_name": "x", "temperature": 1.0, "max_workers": 1},
        "counts": {"n_total": 2, "n_success": 2, "n_fail": 0},
        "outputs": {"success_files": []},
    }
    p4_path = repo / "artifacts" / "cache" / "pipeline4_manifest.json"
    _write_json(p4_path, p4)
    return p4_path


def _make_pipeline5_manifest(
    repo: Path,
    *,
    jsonl_path: Path,
    p2_path: Path | None,
    p4_path: Path | None,
) -> Path:
    p5 = {
        "meta": {
            "pipeline": 5,
            "outputs_jsonl": str(jsonl_path),
            "outputs_psv": str(repo / "artifacts" / "outputs" / "output.psv"),
            "pipeline4_manifest_path": str(p4_path) if p4_path else None,
            "pipeline2_ingest_path": str(p2_path) if p2_path else None,
        },
        "counts": {"n_exported": 2},
    }
    p5_path = repo / "artifacts" / "cache" / "pipeline5_manifest.json"
    _write_json(p5_path, p5)
    return p5_path


@pytest.mark.parametrize("write_html", [True, False])
def test_pipeline_6_writes_report_and_manifest(tmp_path: Path, write_html: bool) -> None:
    repo = tmp_path
    _make_repo_skeleton(repo)

    params_yaml = f"""
run:
  log_level: INFO
input:
  path: raw_data/input.csv
  format: csv
report:
  enabled: true
  md_path: artifacts/reports/report.md
  html_path: artifacts/reports/report.html
  write_html: {str(write_html).lower()}
  sample_per_group: 2
  include_full_examples: false
  max_reason_examples: 5
"""
    _write(repo / "configs" / "parameters.yaml", params_yaml)

    p2_path = _make_min_pipeline2_input(repo, n_rows=2)
    p4_path = _make_min_pipeline4_manifest(repo, judge_enabled=True)

    jsonl_path = repo / "artifacts" / "outputs" / "output.jsonl"
    records = [
        {
            "input": {"Role Track Example Name": "Data Scientist", "Question Set": "A", "Question ID": "1", "Question Type": "Generic"},
            "parsed": {"question_name": "Q1", "example_answer_good": "g", "example_answer_mid": "m", "example_answer_bad": "b", "grading_rubrics": "r"},
            "judge": {"verdict": "PASS", "score": 90, "reasons": ["ok"]},
            "meta": {"cache_id": "c1"},
        },
        {
            "input": {"Role Track Example Name": "Product Owner", "Question Set": "B", "Question ID": "2", "Question Type": "Specific"},
            "parsed": {"question_name": "Q2", "example_answer_good": "g", "example_answer_mid": "m", "example_answer_bad": "b", "grading_rubrics": "r"},
            "judge": {"verdict": "FAIL", "score": 10, "reasons": ["bad"]},
            "meta": {"cache_id": "c2"},
        },
    ]
    _write_jsonl(jsonl_path, records)

    p5_path = _make_pipeline5_manifest(repo, jsonl_path=jsonl_path, p2_path=p2_path, p4_path=p4_path)

    rc = pipeline6_main(
        parameters_path=str(repo / "configs" / "parameters.yaml"),
        pipeline5_manifest_path=str(p5_path),
        out_manifest_path=str(repo / "artifacts" / "cache" / "pipeline6_manifest.json"),
    )
    assert rc == 0

    md_out = repo / "artifacts" / "reports" / "report.md"
    assert md_out.exists()
    md_text = md_out.read_text(encoding="utf-8")
    assert "Pipeline 6 — Report" in md_text
    assert "Run Summary" in md_text
    assert "Judge" in md_text

    html_out = repo / "artifacts" / "reports" / "report.html"
    if write_html:
        assert html_out.exists()
    else:
        assert not html_out.exists()

    m6 = repo / "artifacts" / "cache" / "pipeline6_manifest.json"
    assert m6.exists()
    manifest = json.loads(m6.read_text(encoding="utf-8"))
    assert manifest["counts"]["n_records_jsonl"] == 2
    assert manifest["counts"]["n_pass"] == 1
    assert manifest["counts"]["n_fail"] == 1
    assert manifest["counts"]["judge_enabled"] is True


def test_pipeline_6_skips_when_report_disabled(tmp_path: Path) -> None:
    repo = tmp_path
    _make_repo_skeleton(repo)

    params_yaml = """
run:
  log_level: INFO
input:
  path: raw_data/input.csv
  format: csv
report:
  enabled: false
  md_path: artifacts/reports/report.md
  html_path: artifacts/reports/report.html
  write_html: true
"""
    _write(repo / "configs" / "parameters.yaml", params_yaml)

    # Even if pipeline5 manifest is missing, it should early-exit cleanly.
    rc = pipeline6_main(parameters_path=str(repo / "configs" / "parameters.yaml"))
    assert rc == 0

    assert not (repo / "artifacts" / "reports" / "report.md").exists()
    assert not (repo / "artifacts" / "reports" / "report.html").exists()
    assert not (repo / "artifacts" / "cache" / "pipeline6_manifest.json").exists()


def test_pipeline_6_raises_when_jsonl_missing(tmp_path: Path) -> None:
    repo = tmp_path
    _make_repo_skeleton(repo)

    params_yaml = """
run:
  log_level: INFO
input:
  path: raw_data/input.csv
  format: csv
report:
  enabled: true
  md_path: artifacts/reports/report.md
  html_path: artifacts/reports/report.html
  write_html: false
  sample_per_group: 1
  include_full_examples: false
  max_reason_examples: 5
"""
    _write(repo / "configs" / "parameters.yaml", params_yaml)

    # Pipeline 5 manifest points to a non-existent JSONL
    missing_jsonl = repo / "artifacts" / "outputs" / "does_not_exist.jsonl"
    p2_path = _make_min_pipeline2_input(repo, n_rows=1)
    p4_path = _make_min_pipeline4_manifest(repo, judge_enabled=True)
    p5_path = _make_pipeline5_manifest(repo, jsonl_path=missing_jsonl, p2_path=p2_path, p4_path=p4_path)

    with pytest.raises(FileNotFoundError):
        pipeline6_main(
            parameters_path=str(repo / "configs" / "parameters.yaml"),
            pipeline5_manifest_path=str(p5_path),
            out_manifest_path=str(repo / "artifacts" / "cache" / "pipeline6_manifest.json"),
        )


def test_pipeline_6_raises_on_invalid_jsonl_line(tmp_path: Path) -> None:
    repo = tmp_path
    _make_repo_skeleton(repo)

    params_yaml = """
run:
  log_level: INFO
input:
  path: raw_data/input.csv
  format: csv
report:
  enabled: true
  md_path: artifacts/reports/report.md
  html_path: artifacts/reports/report.html
  write_html: false
  sample_per_group: 1
  include_full_examples: false
  max_reason_examples: 5
"""
    _write(repo / "configs" / "parameters.yaml", params_yaml)

    p2_path = _make_min_pipeline2_input(repo, n_rows=1)
    p4_path = _make_min_pipeline4_manifest(repo, judge_enabled=True)

    jsonl_path = repo / "artifacts" / "outputs" / "output.jsonl"
    _write(jsonl_path, '{"input": {"Role Track Example Name": "X"}}\n{NOT_JSON}\n')

    p5_path = _make_pipeline5_manifest(repo, jsonl_path=jsonl_path, p2_path=p2_path, p4_path=p4_path)

    with pytest.raises(ValueError):
        pipeline6_main(
            parameters_path=str(repo / "configs" / "parameters.yaml"),
            pipeline5_manifest_path=str(p5_path),
            out_manifest_path=str(repo / "artifacts" / "cache" / "pipeline6_manifest.json"),
        )


def test_pipeline_6_runs_without_pipeline4_manifest_and_no_judge(tmp_path: Path) -> None:
    """
    Ensure pipeline6 still works when:
    - pipeline4_manifest_path is missing/None
    - JSONL records have no judge objects
    """
    repo = tmp_path
    _make_repo_skeleton(repo)

    params_yaml = """
run:
  log_level: INFO
input:
  path: raw_data/input.csv
  format: csv
report:
  enabled: true
  md_path: artifacts/reports/report.md
  html_path: artifacts/reports/report.html
  write_html: false
  sample_per_group: 1
  include_full_examples: false
  max_reason_examples: 5
"""
    _write(repo / "configs" / "parameters.yaml", params_yaml)

    p2_path = _make_min_pipeline2_input(repo, n_rows=2)

    jsonl_path = repo / "artifacts" / "outputs" / "output.jsonl"
    records = [
        {
            "input": {"Role Track Example Name": "Software Developer", "Question Set": "A", "Question ID": "1", "Question Type": "Generic"},
            "parsed": {"question_name": "Q1"},
            "meta": {"cache_id": "c1"},
        },
        {
            "input": {"Role Track Example Name": "Software Developer", "Question Set": "A", "Question ID": "2", "Question Type": "Specific"},
            "parsed": {"question_name": "Q2"},
            "meta": {"cache_id": "c2"},
        },
    ]
    _write_jsonl(jsonl_path, records)

    # pipeline4_manifest_path intentionally omitted
    p5_path = _make_pipeline5_manifest(repo, jsonl_path=jsonl_path, p2_path=p2_path, p4_path=None)

    rc = pipeline6_main(
        parameters_path=str(repo / "configs" / "parameters.yaml"),
        pipeline5_manifest_path=str(p5_path),
        out_manifest_path=str(repo / "artifacts" / "cache" / "pipeline6_manifest.json"),
    )
    assert rc == 0

    md_out = repo / "artifacts" / "reports" / "report.md"
    assert md_out.exists()

    m6 = repo / "artifacts" / "cache" / "pipeline6_manifest.json"
    assert m6.exists()
    manifest = json.loads(m6.read_text(encoding="utf-8"))

    assert manifest["counts"]["n_records_jsonl"] == 2
    # With no judge objects, compute_report_stats should report judge_enabled false
    assert manifest["counts"]["judge_enabled"] is False
    assert manifest["counts"]["n_pass"] == 0
    assert manifest["counts"]["n_fail"] == 0
