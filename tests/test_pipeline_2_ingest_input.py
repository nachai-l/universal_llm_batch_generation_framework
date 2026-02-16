# tests/test_pipeline_2_ingest_input.py

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

import functions.batch.pipeline_2_ingest_input as p2


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _mk_params(*, input_path: str, required_columns=None, cache_dir: str = "artifacts/cache"):
    return SimpleNamespace(
        input=SimpleNamespace(
            path=input_path,
            format="csv",
            encoding="utf-8",
            sheet=None,
            required_columns=required_columns,
        ),
        cache=SimpleNamespace(dir=cache_dir),
        # ensure_dirs touches these; we monkeypatch ensure_dirs to no-op
        artifacts=SimpleNamespace(outputs_dir="artifacts/outputs", reports_dir="artifacts/reports", logs_dir="artifacts/logs"),
        outputs=SimpleNamespace(psv_path="artifacts/outputs/output.psv", jsonl_path="artifacts/outputs/output.jsonl"),
        report=SimpleNamespace(md_path="artifacts/reports/report.md", html_path="artifacts/reports/report.html"),
        llm_schema=SimpleNamespace(archive_dir="archived/", py_path="schema/llm_schema.py", txt_path="schema/llm_schema.txt"),
    )


def test_pipeline_2_writes_json_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    _write(
        tmp_path,
        "raw_data/input.csv",
        "A,B\nhello,\nworld,test\n",
    )

    params = _mk_params(input_path="raw_data/input.csv", required_columns=None, cache_dir="artifacts/cache")

    monkeypatch.setattr(p2, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p2, "ensure_dirs", lambda *a, **k: None)

    assert p2.main() == 0

    out = tmp_path / "artifacts/cache/pipeline2_input.json"
    assert out.exists()

    obj = json.loads(out.read_text(encoding="utf-8"))
    assert "meta" in obj and "rows" in obj
    assert obj["meta"]["n_rows"] == 2
    assert obj["meta"]["columns"] == ["A", "B"]


def test_pipeline_2_required_columns_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    _write(tmp_path, "raw_data/input.csv", "A,B\nx,y\n")

    params = _mk_params(
        input_path="raw_data/input.csv",
        required_columns=["A", "MISSING"],
        cache_dir="artifacts/cache",
    )

    monkeypatch.setattr(p2, "load_parameters", lambda *a, **k: params)
    monkeypatch.setattr(p2, "ensure_dirs", lambda *a, **k: None)

    with pytest.raises(ValueError) as e:
        p2.main()

    assert "missing required columns" in str(e.value).lower()
