# tests/test_pipeline_4_llm_generate.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from functions.core.llm_batch_storage import stable_cache_id


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_minimal_schema_and_prompts(repo: Path) -> tuple[Path, Path]:
    (repo / "schema").mkdir(parents=True, exist_ok=True)
    (repo / "prompts").mkdir(parents=True, exist_ok=True)

    # --- schema/llm_schema.py (minimal) ---
    (repo / "schema" / "llm_schema.py").write_text(
        """
from pydantic import BaseModel, ConfigDict

class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    foo: str

class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    feedback: str = ""
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (repo / "schema" / "llm_schema.txt").write_text('{"type":"object"}\n', encoding="utf-8")

    gen_prompt = repo / "prompts" / "generation.yaml"
    judge_prompt = repo / "prompts" / "judge.yaml"
    gen_prompt.write_text("name: gen\nuser: |\n  {context}\n  {llm_schema}\n", encoding="utf-8")
    judge_prompt.write_text("name: judge\nuser: |\n  {output_json}\n", encoding="utf-8")

    return gen_prompt, judge_prompt


def _write_parameters(repo: Path, *, cache_force: bool) -> None:
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "configs" / "parameters.yaml").write_text(
        f"""
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
    {{__ROW_KV_BLOCK__}}
  auto_kv_block: true
  kv_order: input_order
  max_context_chars: 12000
  truncate_field_chars: 0
  group_header_template: null
  group_footer_template: null

prompts:
  generation:
    path: prompts/generation.yaml
  judge:
    enabled: true
    path: prompts/judge.yaml
  schema_auto_py_generation:
    path: prompts/schema_auto_py_generation.yaml
  schema_auto_json_summarization:
    path: prompts/schema_auto_json_summarization.yaml

llm_schema:
  py_path: schema/llm_schema.py
  txt_path: schema/llm_schema.txt
  auto_generate: false
  force_regenerate: false
  archive_dir: archived/

llm:
  model_name: dummy
  temperature: 0.0
  max_retries: 2
  timeout_sec: 60
  max_workers: 1
  silence_client_lv_logs: true

cache:
  enabled: true
  force: {"true" if cache_force else "false"}
  dir: artifacts/cache
  dump_failures: true

artifacts:
  dir: artifacts
  outputs_dir: artifacts/outputs
  reports_dir: artifacts/reports
  logs_dir: artifacts/logs

outputs:
  formats: [jsonl]
  psv_path: artifacts/outputs/output.psv
  jsonl_path: artifacts/outputs/output.jsonl

report:
  enabled: false
  md_path: artifacts/reports/report.md
  html_path: artifacts/reports/report.html
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_pipeline3_dedup_artifacts(repo: Path, *, n_items: int) -> None:
    (repo / "artifacts" / "cache").mkdir(parents=True, exist_ok=True)

    # IMPORTANT: Pipeline 4 always reads group_contexts_path first (even if later cache-skip),
    # so tests must always write this file.
    _write_json(
        repo / "artifacts" / "cache" / "pipeline3_group_contexts.json",
        {
            "n_groups": 1,
            "groups": [{"group_key": "A", "group_context_id": "gid_A", "context": "q: hello\nq: world"}],
        },
    )

    items = []
    for i in range(n_items):
        items.append(
            {
                "work_id": f"w{i+1}",
                "group_key": "A",
                "row_index": i,
                "group_context_id": "gid_A",
                # DEDUP mode: Pipeline 4 resolves context via gc_map if this flag is True
                "meta": {"deduped_group_context": True, "group_context_id": "gid_A"},
            }
        )

    _write_json(
        repo / "artifacts" / "cache" / "pipeline3_work_items.json",
        {"n_items": n_items, "items": items},
    )


def _monkeypatch_client_factory(monkeypatch):
    import types

    dummy_factory_mod = types.SimpleNamespace(build_gemini_client=lambda silence_logs=True: object())
    monkeypatch.setitem(__import__("sys").modules, "functions.llm.client_factory", dummy_factory_mod)


def test_pipeline4_writes_one_file_per_success(tmp_path, monkeypatch):
    repo = tmp_path
    _write_minimal_schema_and_prompts(repo)
    _write_parameters(repo, cache_force=False)
    _write_pipeline3_dedup_artifacts(repo, n_items=2)
    _monkeypatch_client_factory(monkeypatch)

    # --- monkeypatch runner: gen always OK, judge always pass ---
    from pydantic import BaseModel, ConfigDict

    class _Gen(BaseModel):
        model_config = ConfigDict(extra="forbid")
        foo: str

    class _Judge(BaseModel):
        model_config = ConfigDict(extra="forbid")
        passed: bool
        feedback: str = ""

    def _fake_run_prompt_yaml_json(prompt_path, variables, schema_model, client_ctx, **kwargs):
        if str(prompt_path).endswith("judge.yaml"):
            return _Judge(passed=True, feedback="")
        return _Gen(foo="ok")

    monkeypatch.setattr("functions.batch.pipeline_4_llm_generate.run_prompt_yaml_json", _fake_run_prompt_yaml_json)

    from functions.batch.pipeline_4_llm_generate import main as p4_main

    rc = p4_main(
        parameters_path=repo / "configs" / "parameters.yaml",
        work_items_path=repo / "artifacts" / "cache" / "pipeline3_work_items.json",
        group_contexts_path=repo / "artifacts" / "cache" / "pipeline3_group_contexts.json",
        outputs_dir=repo / "artifacts" / "cache" / "llm_outputs",
        failures_dir=repo / "artifacts" / "cache" / "llm_failures",
        manifest_path=repo / "artifacts" / "cache" / "pipeline4_manifest.json",
    )
    assert rc == 0

    out_dir = repo / "artifacts" / "cache" / "llm_outputs"
    files = sorted([p.name for p in out_dir.glob("*.json")])
    assert len(files) == 2  # one file per WorkItem success

    manifest = json.loads((repo / "artifacts" / "cache" / "pipeline4_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["n_total"] == 2
    assert manifest["counts"]["n_success"] == 2
    assert manifest["counts"]["n_fail"] == 0


def test_pipeline4_judge_fail_triggers_retry_then_success(tmp_path, monkeypatch):
    repo = tmp_path
    _write_minimal_schema_and_prompts(repo)
    _write_parameters(repo, cache_force=True)
    _write_pipeline3_dedup_artifacts(repo, n_items=1)
    _monkeypatch_client_factory(monkeypatch)

    from pydantic import BaseModel, ConfigDict

    class _Gen(BaseModel):
        model_config = ConfigDict(extra="forbid")
        foo: str

    class _Judge(BaseModel):
        model_config = ConfigDict(extra="forbid")
        passed: bool
        feedback: str = ""

    calls = {"gen": 0, "judge": 0}

    def _fake_run(prompt_path, variables, schema_model, client_ctx, **kwargs):
        if str(prompt_path).endswith("judge.yaml"):
            calls["judge"] += 1
            if calls["judge"] == 1:
                return _Judge(passed=False, feedback="Fix output")
            return _Judge(passed=True, feedback="")
        calls["gen"] += 1
        return _Gen(foo=f"ok{calls['gen']}")

    monkeypatch.setattr("functions.batch.pipeline_4_llm_generate.run_prompt_yaml_json", _fake_run)

    from functions.batch.pipeline_4_llm_generate import main as p4_main

    rc = p4_main(
        parameters_path=repo / "configs" / "parameters.yaml",
        work_items_path=repo / "artifacts" / "cache" / "pipeline3_work_items.json",
        group_contexts_path=repo / "artifacts" / "cache" / "pipeline3_group_contexts.json",
        outputs_dir=repo / "artifacts" / "cache" / "llm_outputs",
        failures_dir=repo / "artifacts" / "cache" / "llm_failures",
        manifest_path=repo / "artifacts" / "cache" / "pipeline4_manifest.json",
    )
    assert rc == 0

    # gen called twice due to judge fail
    assert calls["gen"] == 2
    assert calls["judge"] == 2

    out_dir = repo / "artifacts" / "cache" / "llm_outputs"
    files = list(out_dir.glob("*.json"))
    assert len(files) == 1  # one accepted output

    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["parsed"]["foo"] in {"ok1", "ok2"}
    assert payload["judge"]["passed"] is True


def test_pipeline4_skips_existing_output_when_cache_enabled_and_not_force(tmp_path, monkeypatch):
    """
    If artifacts/cache/llm_outputs/{cache_id}.json already exists:
    - cache.enabled=true
    - cache.force=false
    => Pipeline 4 must SKIP calling the LLM runner for that item and count it as cache_skipped.
    """
    repo = tmp_path
    gen_prompt, judge_prompt = _write_minimal_schema_and_prompts(repo)
    _write_parameters(repo, cache_force=False)
    _write_pipeline3_dedup_artifacts(repo, n_items=1)
    _monkeypatch_client_factory(monkeypatch)

    import functions.batch.pipeline_4_llm_generate as p4

    prompt_sha = p4.sha1_file(gen_prompt)
    schema_sha = p4.sha1_file(repo / "schema" / "llm_schema.txt")
    judge_prompt_sha = p4.sha1_file(judge_prompt)

    cache_id = stable_cache_id(
        work_id="w1",
        prompt_sha=prompt_sha,
        schema_sha=schema_sha,
        model_name="dummy",
        temperature=0.0,
        judge_enabled=True,
        judge_prompt_sha=judge_prompt_sha,
    )

    out_dir = repo / "artifacts" / "cache" / "llm_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cache_id}.json"
    out_file.write_text(
        json.dumps(
            {"meta": {"cache_id": cache_id}, "parsed": {"foo": "cached"}, "judge": {"passed": True}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls = {"n": 0}

    def _fake_run(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("run_prompt_yaml_json should not be called when output exists and cache.force=false")

    monkeypatch.setattr("functions.batch.pipeline_4_llm_generate.run_prompt_yaml_json", _fake_run)

    from functions.batch.pipeline_4_llm_generate import main as p4_main

    rc = p4_main(
        parameters_path=repo / "configs" / "parameters.yaml",
        work_items_path=repo / "artifacts" / "cache" / "pipeline3_work_items.json",
        group_contexts_path=repo / "artifacts" / "cache" / "pipeline3_group_contexts.json",
        outputs_dir=out_dir,
        failures_dir=repo / "artifacts" / "cache" / "llm_failures",
        manifest_path=repo / "artifacts" / "cache" / "pipeline4_manifest.json",
    )
    assert rc == 0
    assert calls["n"] == 0  # critical

    manifest = json.loads((repo / "artifacts" / "cache" / "pipeline4_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["n_total"] == 1
    assert manifest["counts"]["n_success"] == 1
    assert manifest["counts"]["n_fail"] == 0
    assert manifest["counts"]["n_cache_skipped"] == 1
    assert cache_id + ".json" in manifest["outputs"]["success_files"]


def test_pipeline4_force_true_overwrites_existing_output(tmp_path, monkeypatch):
    """
    If output file already exists but cache.force=true, Pipeline 4 must re-run generation/judge
    and write a NEW payload (overwrite), i.e. no cache_skipped.
    """
    repo = tmp_path
    gen_prompt, judge_prompt = _write_minimal_schema_and_prompts(repo)
    _write_parameters(repo, cache_force=True)
    _write_pipeline3_dedup_artifacts(repo, n_items=1)
    _monkeypatch_client_factory(monkeypatch)

    import functions.batch.pipeline_4_llm_generate as p4

    prompt_sha = p4.sha1_file(gen_prompt)
    schema_sha = p4.sha1_file(repo / "schema" / "llm_schema.txt")
    judge_prompt_sha = p4.sha1_file(judge_prompt)

    cache_id = stable_cache_id(
        work_id="w1",
        prompt_sha=prompt_sha,
        schema_sha=schema_sha,
        model_name="dummy",
        temperature=0.0,
        judge_enabled=True,
        judge_prompt_sha=judge_prompt_sha,
    )

    out_dir = repo / "artifacts" / "cache" / "llm_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cache_id}.json"

    out_file.write_text(
        json.dumps({"meta": {"cache_id": cache_id}, "parsed": {"foo": "OLD"}, "judge": {"passed": True}}, ensure_ascii=False),
        encoding="utf-8",
    )

    from pydantic import BaseModel, ConfigDict

    class _Gen(BaseModel):
        model_config = ConfigDict(extra="forbid")
        foo: str

    class _Judge(BaseModel):
        model_config = ConfigDict(extra="forbid")
        passed: bool
        feedback: str = ""

    calls = {"gen": 0, "judge": 0}

    def _fake_run(prompt_path, variables, schema_model, client_ctx, **kwargs):
        if str(prompt_path).endswith("judge.yaml"):
            calls["judge"] += 1
            return _Judge(passed=True, feedback="")
        calls["gen"] += 1
        return _Gen(foo="NEW")

    monkeypatch.setattr("functions.batch.pipeline_4_llm_generate.run_prompt_yaml_json", _fake_run)

    from functions.batch.pipeline_4_llm_generate import main as p4_main

    rc = p4_main(
        parameters_path=repo / "configs" / "parameters.yaml",
        work_items_path=repo / "artifacts" / "cache" / "pipeline3_work_items.json",
        group_contexts_path=repo / "artifacts" / "cache" / "pipeline3_group_contexts.json",
        outputs_dir=out_dir,
        failures_dir=repo / "artifacts" / "cache" / "llm_failures",
        manifest_path=repo / "artifacts" / "cache" / "pipeline4_manifest.json",
    )
    assert rc == 0
    assert calls["gen"] == 1
    assert calls["judge"] == 1

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["parsed"]["foo"] == "NEW"

    manifest = json.loads((repo / "artifacts" / "cache" / "pipeline4_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["n_total"] == 1
    assert manifest["counts"]["n_success"] == 1
    assert manifest["counts"]["n_fail"] == 0
    assert manifest["counts"]["n_cache_skipped"] == 0
