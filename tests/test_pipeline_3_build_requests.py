# tests/test_pipeline_3_build_requests.py

from __future__ import annotations

import json
from pathlib import Path

from functions.batch.pipeline_3_build_requests import main as pipeline3_main


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_pipeline3_build_requests_writes_items_and_group_contexts_dedup(tmp_path):
    """
    Integration-ish test (DEDUP mode):
    - grouping.enabled=true + mode=row_output_with_group_context triggers:
        - pipeline3_work_items.json (items with empty context + group_context_id in meta)
        - pipeline3_group_contexts.json (unique group contexts)
    """
    repo = tmp_path
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "cache").mkdir(parents=True, exist_ok=True)

    (repo / "configs" / "parameters.yaml").write_text(
        """
run:
  name: t
  timezone: Asia/Tokyo
  log_level: INFO

input:
  path: raw_data/input.csv
  format: csv
  encoding: utf-8
  sheet: null
  required_columns: null

grouping:
  enabled: true
  column: "group"
  mode: row_output_with_group_context
  max_rows_per_group: 50

context:
  columns:
    mode: all
    include: []
    exclude: []
  row_template: |
    {__ROW_KV_BLOCK__}
  auto_kv_block: true
  kv_order: input_order
  max_context_chars: 12000
  truncate_field_chars: 0
  group_header_template: null
  group_footer_template: null
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ingest_path = repo / "artifacts" / "cache" / "pipeline2_input.json"
    _write_json(
        ingest_path,
        {
            "meta": {"columns": ["group", "q"], "n_rows": 3, "n_cols": 2, "input_path": "x.csv"},
            "records": [
                {"group": "A", "q": "hello"},
                {"group": "A", "q": "world"},
                {"group": "B", "q": "foo"},
            ],
        },
    )

    out_items_path = repo / "artifacts" / "cache" / "pipeline3_work_items.json"
    out_groups_path = repo / "artifacts" / "cache" / "pipeline3_group_contexts.json"

    rc = pipeline3_main(
        parameters_path=repo / "configs" / "parameters.yaml",
        ingest_path=ingest_path,
        out_path=out_items_path,
        out_group_contexts_path=out_groups_path,
    )
    assert rc == 0
    assert out_items_path.exists()
    assert out_groups_path.exists()

    out = json.loads(out_items_path.read_text(encoding="utf-8"))
    assert out["n_items"] == 3
    assert len(out["items"]) == 3

    # Dedup metadata markers
    assert out["meta"]["grouping"]["enabled"] is True
    assert out["meta"]["grouping"]["mode"] == "row_output_with_group_context"
    assert out["meta"]["grouping"]["dedupe_group_context"] is True
    assert out["n_group_contexts"] == 2
    assert "group_contexts_artifact" in out

    # items should NOT repeat context
    one = out["items"][0]
    assert set(one.keys()) >= {"work_id", "group_key", "row_index", "context", "meta"}
    assert isinstance(one["work_id"], str) and len(one["work_id"]) == 40  # sha1 hex
    assert one["group_key"] in {"A", "B"}
    assert one["row_index"] in {0, 1, 2}
    assert one["context"] == ""  # dedup mode empties context
    assert isinstance(one["meta"], dict)
    assert "group_context_id" in one["meta"]
    assert isinstance(one["meta"]["group_context_id"], str) and len(one["meta"]["group_context_id"]) == 40

    # group contexts artifact should contain exactly 2 unique groups
    groups = json.loads(out_groups_path.read_text(encoding="utf-8"))
    assert isinstance(groups, list)
    assert len(groups) == 2
    gkeys = {g["group_key"] for g in groups}
    assert gkeys == {"A", "B"}
    for g in groups:
        assert set(g.keys()) >= {"group_key", "group_context_id", "context", "meta"}
        assert isinstance(g["group_context_id"], str) and len(g["group_context_id"]) == 40
        assert isinstance(g["context"], str) and len(g["context"]) > 0


def test_pipeline3_build_requests_non_dedup_rowwise(tmp_path):
    """
    Non-dedup path (grouping.enabled=false):
    - Should write only pipeline3_work_items.json
    - Each WorkItem.context is populated (legacy behavior)
    - No pipeline3_group_contexts.json is created
    """
    repo = tmp_path
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "cache").mkdir(parents=True, exist_ok=True)

    (repo / "configs" / "parameters.yaml").write_text(
        """
run:
  name: t
  timezone: Asia/Tokyo
  log_level: INFO

input:
  path: raw_data/input.csv
  format: csv
  encoding: utf-8
  sheet: null
  required_columns: null

grouping:
  enabled: false
  column: null
  mode: group_output
  max_rows_per_group: 50

context:
  columns:
    mode: all
    include: []
    exclude: []
  row_template: |
    {__ROW_KV_BLOCK__}
  auto_kv_block: true
  kv_order: input_order
  max_context_chars: 12000
  truncate_field_chars: 0
  group_header_template: null
  group_footer_template: null
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ingest_path = repo / "artifacts" / "cache" / "pipeline2_input.json"
    _write_json(
        ingest_path,
        {
            "meta": {"columns": ["group", "q"], "n_rows": 2, "n_cols": 2, "input_path": "x.csv"},
            "records": [
                {"group": "A", "q": "hello"},
                {"group": "B", "q": "foo"},
            ],
        },
    )

    out_items_path = repo / "artifacts" / "cache" / "pipeline3_work_items.json"
    out_groups_path = repo / "artifacts" / "cache" / "pipeline3_group_contexts.json"

    rc = pipeline3_main(
        parameters_path=repo / "configs" / "parameters.yaml",
        ingest_path=ingest_path,
        out_path=out_items_path,
        out_group_contexts_path=out_groups_path,
    )
    assert rc == 0
    assert out_items_path.exists()
    assert not out_groups_path.exists()

    out = json.loads(out_items_path.read_text(encoding="utf-8"))
    assert out["n_items"] == 2
    assert len(out["items"]) == 2

    # Non-dedup markers
    assert out["meta"]["grouping"]["dedupe_group_context"] is False
    assert "n_group_contexts" not in out
    assert "group_contexts_artifact" not in out

    one = out["items"][0]
    assert isinstance(one["context"], str) and len(one["context"]) > 0
    assert one["group_key"] == ""  # grouping disabled => no group_key


def test_pipeline3_empty_records_produces_empty_items(tmp_path):
    repo = tmp_path
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "cache").mkdir(parents=True, exist_ok=True)

    (repo / "configs" / "parameters.yaml").write_text(
        """
run:
  name: t
  timezone: Asia/Tokyo
  log_level: INFO

input:
  path: raw_data/input.csv
  format: csv
  encoding: utf-8
  sheet: null
  required_columns: null

grouping:
  enabled: false
  column: null
  mode: group_output
  max_rows_per_group: 50

context:
  columns:
    mode: all
    include: []
    exclude: []
  row_template: |
    {__ROW_KV_BLOCK__}
  auto_kv_block: true
  kv_order: input_order
  max_context_chars: 12000
  truncate_field_chars: 0
  group_header_template: null
  group_footer_template: null
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ingest_path = repo / "artifacts" / "cache" / "pipeline2_input.json"
    _write_json(ingest_path, {"meta": {"n_rows": 0, "n_cols": 0, "columns": []}, "records": []})

    out_items_path = repo / "artifacts" / "cache" / "pipeline3_work_items.json"
    out_groups_path = repo / "artifacts" / "cache" / "pipeline3_group_contexts.json"

    rc = pipeline3_main(
        parameters_path=repo / "configs" / "parameters.yaml",
        ingest_path=ingest_path,
        out_path=out_items_path,
        out_group_contexts_path=out_groups_path,
    )
    assert rc == 0
    assert out_items_path.exists()
    assert not out_groups_path.exists()

    out = json.loads(out_items_path.read_text(encoding="utf-8"))
    assert out["n_items"] == 0
    assert out["items"] == []


# ✅ NEW CASE 1: grouping.enabled=true + mode=group_output should NOT write group_contexts file,
# and each group output WorkItem should include meta.group_context_id.
def test_pipeline3_group_output_writes_items_only_and_has_group_context_id(tmp_path):
    repo = tmp_path
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "cache").mkdir(parents=True, exist_ok=True)

    (repo / "configs" / "parameters.yaml").write_text(
        """
run:
  name: t
  timezone: Asia/Tokyo
  log_level: INFO

input:
  path: raw_data/input.csv
  format: csv
  encoding: utf-8
  sheet: null
  required_columns: null

grouping:
  enabled: true
  column: "group"
  mode: group_output
  max_rows_per_group: 50

context:
  columns:
    mode: all
    include: []
    exclude: []
  row_template: |
    {__ROW_KV_BLOCK__}
  auto_kv_block: true
  kv_order: input_order
  max_context_chars: 12000
  truncate_field_chars: 0
  group_header_template: null
  group_footer_template: null
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ingest_path = repo / "artifacts" / "cache" / "pipeline2_input.json"
    _write_json(
        ingest_path,
        {
            "meta": {"columns": ["group", "q"], "n_rows": 3, "n_cols": 2, "input_path": "x.csv"},
            "records": [
                {"group": "A", "q": "hello"},
                {"group": "A", "q": "world"},
                {"group": "B", "q": "foo"},
            ],
        },
    )

    out_items_path = repo / "artifacts" / "cache" / "pipeline3_work_items.json"
    out_groups_path = repo / "artifacts" / "cache" / "pipeline3_group_contexts.json"

    rc = pipeline3_main(
        parameters_path=repo / "configs" / "parameters.yaml",
        ingest_path=ingest_path,
        out_path=out_items_path,
        out_group_contexts_path=out_groups_path,
    )
    assert rc == 0
    assert out_items_path.exists()
    assert not out_groups_path.exists()

    out = json.loads(out_items_path.read_text(encoding="utf-8"))
    assert out["n_items"] == 2  # groups A,B
    assert out["meta"]["grouping"]["enabled"] is True
    assert out["meta"]["grouping"]["mode"] == "group_output"
    assert out["meta"]["grouping"]["dedupe_group_context"] is False
    assert "n_group_contexts" not in out

    # Each item is a group output and should have group_context_id + non-empty context
    for it in out["items"]:
        assert it["row_index"] is None
        assert it["group_key"] in {"A", "B"}
        assert isinstance(it["context"], str) and len(it["context"]) > 0
        assert isinstance(it["meta"], dict)
        assert isinstance(it["meta"].get("group_context_id"), str) and len(it["meta"]["group_context_id"]) == 40


# ✅ NEW CASE 2: dedup flag should depend ONLY on mode=row_output_with_group_context.
# grouping.enabled=true but mode=group_output => still non-dedup.
def test_pipeline3_grouping_enabled_but_not_row_output_mode_is_not_dedup(tmp_path):
    repo = tmp_path
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "cache").mkdir(parents=True, exist_ok=True)

    (repo / "configs" / "parameters.yaml").write_text(
        """
run:
  name: t
  timezone: Asia/Tokyo
  log_level: INFO

input:
  path: raw_data/input.csv
  format: csv
  encoding: utf-8
  sheet: null
  required_columns: null

grouping:
  enabled: true
  column: "group"
  mode: group_output
  max_rows_per_group: 50

context:
  columns:
    mode: all
    include: []
    exclude: []
  row_template: |
    {__ROW_KV_BLOCK__}
  auto_kv_block: true
  kv_order: input_order
  max_context_chars: 12000
  truncate_field_chars: 0
  group_header_template: null
  group_footer_template: null
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ingest_path = repo / "artifacts" / "cache" / "pipeline2_input.json"
    _write_json(
        ingest_path,
        {
            "meta": {"columns": ["group", "q"], "n_rows": 1, "n_cols": 2, "input_path": "x.csv"},
            "records": [{"group": "A", "q": "hello"}],
        },
    )

    out_items_path = repo / "artifacts" / "cache" / "pipeline3_work_items.json"
    out_groups_path = repo / "artifacts" / "cache" / "pipeline3_group_contexts.json"

    rc = pipeline3_main(
        parameters_path=repo / "configs" / "parameters.yaml",
        ingest_path=ingest_path,
        out_path=out_items_path,
        out_group_contexts_path=out_groups_path,
    )
    assert rc == 0
    out = json.loads(out_items_path.read_text(encoding="utf-8"))
    assert out["meta"]["grouping"]["dedupe_group_context"] is False
    assert not out_groups_path.exists()
