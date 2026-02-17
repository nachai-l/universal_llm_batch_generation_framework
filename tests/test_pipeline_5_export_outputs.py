# tests/test_pipeline_5_export_outputs.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from functions.batch.pipeline_5_export_outputs import main as pipeline5_main


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _get_parsed_question_name(rec: Dict[str, Any]) -> Optional[str]:
    """
    Pipeline 5 JSONL is expected to keep parsed nested (rec["parsed"]).
    But we keep a tolerant helper in case a future refactor flattens JSONL.
    """
    if isinstance(rec.get("parsed"), dict) and "question_name" in rec["parsed"]:
        return rec["parsed"]["question_name"]
    if "question_name" in rec:
        return rec["question_name"]
    return None


def _get_judge_verdict(rec: Dict[str, Any]) -> Optional[str]:
    """
    Similar tolerance for judge: prefer nested judge, else flattened field.
    """
    j = rec.get("judge")
    if isinstance(j, dict) and "verdict" in j:
        return j["verdict"]
    if "judge_verdict" in rec:
        return rec["judge_verdict"]
    return None


# ------------------------------------------------------------
# 1) Row-output mode behavior
# - row_index present => row-wise join to pipeline2_input.records[row_index]
# - grouping.mode must be one of the allowed literals, so we use row_output_with_group_context
# ------------------------------------------------------------

def test_pipeline_5_exports_jsonl_and_psv_row_mode(tmp_path: Path) -> None:
    repo = tmp_path

    params_path = repo / "configs" / "parameters.yaml"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(
        """
run:
  name: test
  timezone: Asia/Tokyo
  log_level: INFO

input:
  path: raw_data/input.csv
  format: csv

grouping:
  enabled: false
  column: null
  mode: row_output_with_group_context
  max_rows_per_group: 50

outputs:
  formats: [psv, jsonl]
  psv_path: artifacts/outputs/output.psv
  jsonl_path: artifacts/outputs/output.jsonl

artifacts:
  dir: artifacts
""".strip()
        + "\n",
        encoding="utf-8",
    )

    # pipeline2
    p2 = {
        "meta": {"columns": ["Role Track Example Name", "Question ID"]},
        "records": [
            {"Role Track Example Name": "A", "Question ID": "1"},
            {"Role Track Example Name": "B", "Question ID": "2"},
        ],
    }
    p2_path = repo / "artifacts" / "cache" / "pipeline2_input.json"
    _write_json(p2_path, p2)

    # llm_outputs (row-wise)
    out_dir = repo / "artifacts" / "cache" / "llm_outputs"
    _write_json(
        out_dir / "a.json",
        {
            "meta": {"row_index": 0, "work_id": "w1", "cache_id": "a"},
            "parsed": {"question_name": "Q1"},
            "judge": {"verdict": "PASS", "score": 90, "reasons": []},
        },
    )
    _write_json(
        out_dir / "b.json",
        {
            "meta": {"row_index": 1, "work_id": "w2", "cache_id": "b"},
            "parsed": {"question_name": "Q2"},
            "judge": {"verdict": "PASS", "score": 95, "reasons": []},
        },
    )

    p4 = {"outputs": {"success_files": ["b.json", "a.json"]}}
    p4_path = repo / "artifacts" / "cache" / "pipeline4_manifest.json"
    _write_json(p4_path, p4)

    rc = pipeline5_main(
        parameters_path=str(params_path),
        pipeline2_ingest_path=str(p2_path),
        pipeline4_manifest_path=str(p4_path),
        pipeline4_outputs_dir=str(out_dir),
        pipeline5_manifest_path=str(repo / "artifacts" / "cache" / "pipeline5_manifest.json"),
    )
    assert rc == 0

    # Validate JSONL ordering by row_index (0 then 1)
    jsonl_path = repo / "artifacts" / "outputs" / "output.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])

    assert rec0["meta"]["row_index"] == 0
    assert rec1["meta"]["row_index"] == 1

    # Row-wise join is preserved
    assert rec0["input"]["Role Track Example Name"] == "A"
    assert rec1["input"]["Role Track Example Name"] == "B"

    # JSONL: parsed stays nested (expected)
    assert _get_parsed_question_name(rec0) == "Q1"
    assert _get_parsed_question_name(rec1) == "Q2"

    # Judge verdict available (nested preferred)
    assert _get_judge_verdict(rec0) == "PASS"
    assert _get_judge_verdict(rec1) == "PASS"

    # Validate PSV exists and has key columns (PSV is flattened)
    psv_path = repo / "artifacts" / "outputs" / "output.psv"
    df = pd.read_csv(psv_path, sep="|", dtype=str, keep_default_na=False)
    assert df.shape[0] == 2
    assert df.loc[0, "Role Track Example Name"] == "A"
    assert df.loc[1, "Role Track Example Name"] == "B"
    assert df.loc[0, "Question ID"] == "1"
    assert df.loc[1, "Question ID"] == "2"
    assert df.loc[0, "question_name"] == "Q1"
    assert df.loc[1, "question_name"] == "Q2"
    assert df.loc[0, "judge_verdict"] == "PASS"
    assert df.loc[1, "judge_verdict"] == "PASS"


# ------------------------------------------------------------
# 2) Group-output mode expansion
# Expectation (current pipeline 5 behavior from logs):
# - group_outputs=1
# - expanded_rows=2
# - JSONL exported rows are still in the standard structure:
#     input + parsed + judge + meta
#   (NOT flattened in JSONL)
# ------------------------------------------------------------

def test_pipeline_5_group_output_expands_rows(tmp_path: Path) -> None:
    repo = tmp_path

    params_path = repo / "configs" / "parameters.yaml"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(
        """
run:
  name: test
  timezone: Asia/Tokyo
  log_level: INFO

input:
  path: raw_data/input.csv
  format: csv

grouping:
  enabled: true
  column: Role Track Example Name
  mode: group_output
  max_rows_per_group: 50

outputs:
  formats: [psv, jsonl]
  psv_path: artifacts/outputs/output.psv
  jsonl_path: artifacts/outputs/output.jsonl

artifacts:
  dir: artifacts
""".strip()
        + "\n",
        encoding="utf-8",
    )

    # pipeline2 (2 rows in the same group)
    p2 = {
        "meta": {"columns": ["Role Track Example Name", "Question ID"]},
        "records": [
            {"Role Track Example Name": "Consultant", "Question ID": "1"},
            {"Role Track Example Name": "Consultant", "Question ID": "2"},
        ],
    }
    p2_path = repo / "artifacts" / "cache" / "pipeline2_input.json"
    _write_json(p2_path, p2)

    # pipeline3 group context (ties group_context_id -> row indices)
    p3 = [
        {
            "group_context_id": "g1",
            "group_key": "Consultant",
            "context": "some context",
            "meta": {
                "grouping_column": "Role Track Example Name",
                "row_indices_used": [0, 1],
            },
        }
    ]
    p3_path = repo / "artifacts" / "cache" / "pipeline3_group_contexts.json"
    _write_json(p3_path, p3)

    # llm_output (group-level)
    out_dir = repo / "artifacts" / "cache" / "llm_outputs"
    _write_json(
        out_dir / "group.json",
        {
            "meta": {"group_context_id": "g1", "group_key": "Consultant", "row_index": None, "work_id": "wg"},
            "parsed": {
                "questions": [
                    {"question_id": "1", "question_name": "Q1"},
                    {"question_id": "2", "question_name": "Q2"},
                ]
            },
            "judge": {"verdict": "PASS", "score": 92, "reasons": ["ok"]},
        },
    )

    p4 = {"outputs": {"success_files": ["group.json"]}}
    p4_path = repo / "artifacts" / "cache" / "pipeline4_manifest.json"
    _write_json(p4_path, p4)

    rc = pipeline5_main(
        parameters_path=str(params_path),
        pipeline2_ingest_path=str(p2_path),
        pipeline3_group_contexts_path=str(p3_path),
        pipeline4_manifest_path=str(p4_path),
        pipeline4_outputs_dir=str(out_dir),
        pipeline5_manifest_path=str(repo / "artifacts" / "cache" / "pipeline5_manifest.json"),
    )
    assert rc == 0

    jsonl_lines = (repo / "artifacts" / "outputs" / "output.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == 2

    r0 = json.loads(jsonl_lines[0])
    r1 = json.loads(jsonl_lines[1])

    # Expanded rows should join to concrete pipeline2 input rows
    assert r0["input"]["Question ID"] == "1"
    assert r1["input"]["Question ID"] == "2"

    # JSONL: parsed stays nested
    assert _get_parsed_question_name(r0) == "Q1"
    assert _get_parsed_question_name(r1) == "Q2"

    # Judge should be duplicated onto each expanded row (nested)
    assert _get_judge_verdict(r0) == "PASS"
    assert _get_judge_verdict(r1) == "PASS"

    # PSV sanity (flattened)
    df = pd.read_csv(repo / "artifacts" / "outputs" / "output.psv", sep="|", dtype=str, keep_default_na=False)
    assert df.shape[0] == 2
    assert list(df["Question ID"]) == ["1", "2"]
    assert list(df["question_name"]) == ["Q1", "Q2"]
    assert list(df["judge_verdict"]) == ["PASS", "PASS"]


# ------------------------------------------------------------
# 3) Missing group context fallback
# Expectation:
# - Pipeline 5 should still export ONE record (group-level) rather than crashing
# - input will be None (no join), but meta is preserved
# ------------------------------------------------------------

def test_pipeline_5_group_output_missing_context_fallback(tmp_path: Path) -> None:
    repo = tmp_path

    params_path = repo / "configs" / "parameters.yaml"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(
        """
run:
  name: test
  timezone: Asia/Tokyo
  log_level: INFO

input:
  path: raw_data/input.csv
  format: csv

grouping:
  enabled: true
  column: Role Track Example Name
  mode: group_output
  max_rows_per_group: 50

outputs:
  formats: [psv, jsonl]
  psv_path: artifacts/outputs/output.psv
  jsonl_path: artifacts/outputs/output.jsonl

artifacts:
  dir: artifacts
""".strip()
        + "\n",
        encoding="utf-8",
    )

    p2 = {"meta": {"columns": ["Role Track Example Name"]}, "records": []}
    p2_path = repo / "artifacts" / "cache" / "pipeline2_input.json"
    _write_json(p2_path, p2)

    out_dir = repo / "artifacts" / "cache" / "llm_outputs"
    _write_json(
        out_dir / "group.json",
        {
            "meta": {"group_context_id": "missing", "group_key": "X", "row_index": None},
            "parsed": {"questions": []},
            "judge": {"verdict": "PASS", "score": 1, "reasons": []},
        },
    )

    p4 = {"outputs": {"success_files": ["group.json"]}}
    p4_path = repo / "artifacts" / "cache" / "pipeline4_manifest.json"
    _write_json(p4_path, p4)

    rc = pipeline5_main(
        parameters_path=str(params_path),
        pipeline2_ingest_path=str(p2_path),
        pipeline4_manifest_path=str(p4_path),
        pipeline4_outputs_dir=str(out_dir),
        pipeline5_manifest_path=str(repo / "artifacts" / "cache" / "pipeline5_manifest.json"),
    )
    assert rc == 0

    jsonl_lines = (repo / "artifacts" / "outputs" / "output.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == 1
    rec = json.loads(jsonl_lines[0])

    # No row-wise join possible
    assert rec.get("input") is None
    assert isinstance(rec.get("meta"), dict)
    assert rec["meta"].get("group_key") == "X"


# ------------------------------------------------------------
# 4) Row-output: out-of-range row_index should not crash; input becomes None
# ------------------------------------------------------------

def test_pipeline_5_row_output_row_index_oob_sets_input_none(tmp_path: Path) -> None:
    repo = tmp_path

    params_path = repo / "configs" / "parameters.yaml"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(
        """
run:
  name: test
  timezone: Asia/Tokyo
  log_level: INFO

input:
  path: raw_data/input.csv
  format: csv

grouping:
  enabled: false
  column: null
  mode: row_output_with_group_context
  max_rows_per_group: 50

outputs:
  formats: [jsonl]
  jsonl_path: artifacts/outputs/output.jsonl
  psv_path: artifacts/outputs/output.psv

artifacts:
  dir: artifacts
""".strip()
        + "\n",
        encoding="utf-8",
    )

    # pipeline2 only has 1 record
    p2 = {
        "meta": {"columns": ["Role Track Example Name", "Question ID"]},
        "records": [{"Role Track Example Name": "A", "Question ID": "1"}],
    }
    p2_path = repo / "artifacts" / "cache" / "pipeline2_input.json"
    _write_json(p2_path, p2)

    # llm_output references row_index 999 (oob)
    out_dir = repo / "artifacts" / "cache" / "llm_outputs"
    _write_json(
        out_dir / "oob.json",
        {
            "meta": {"row_index": 999, "work_id": "w", "cache_id": "oob"},
            "parsed": {"question_name": "Q"},
            "judge": {"verdict": "PASS", "score": 1, "reasons": []},
        },
    )

    p4 = {"outputs": {"success_files": ["oob.json"]}}
    p4_path = repo / "artifacts" / "cache" / "pipeline4_manifest.json"
    _write_json(p4_path, p4)

    rc = pipeline5_main(
        parameters_path=str(params_path),
        pipeline2_ingest_path=str(p2_path),
        pipeline4_manifest_path=str(p4_path),
        pipeline4_outputs_dir=str(out_dir),
        pipeline5_manifest_path=str(repo / "artifacts" / "cache" / "pipeline5_manifest.json"),
    )
    assert rc == 0

    jsonl_lines = (repo / "artifacts" / "outputs" / "output.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == 1
    rec = json.loads(jsonl_lines[0])
    assert rec.get("input") is None
    assert rec["meta"]["row_index"] == 999


import pandas as pd

from functions.core.export_outputs import flatten_for_psv


def test_flatten_for_psv_semantic_collision_question_id_is_prefixed():
    """
    If input has 'Question ID' and parsed has 'question_id', we must NOT overwrite.
    Parsed should become 'parsed_question_id'.
    """
    rec = {
        "input": {"Question ID": "3", "Role": "Data Scientist"},
        "parsed": {"question_id": "3", "question_name": "X"},
        "judge": None,
        "meta": {"row_index": 0},
    }

    out = flatten_for_psv(rec, input_columns=["Role", "Question ID"])

    assert out["Question ID"] == "3"
    assert out["parsed_question_id"] == "3"
    assert out["question_name"] == "X"


def test_flatten_for_psv_exact_collision_is_prefixed():
    """
    If parsed key exactly collides with an existing key in the output dict,
    it must be prefixed (parsed_*) to avoid overwriting.
    """
    rec = {
        "input": {"Role": "Data Scientist", "Question ID": "1"},
        "parsed": {"Role": "SHOULD_NOT_OVERWRITE_INPUT"},
        "judge": None,
        "meta": {"row_index": 0},
    }

    out = flatten_for_psv(rec, input_columns=["Role", "Question ID"])

    assert out["Role"] == "Data Scientist"
    assert out["parsed_Role"] == "SHOULD_NOT_OVERWRITE_INPUT"
