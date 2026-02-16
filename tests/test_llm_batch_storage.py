# tests/test_llm_batch_storage.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from functions.core.llm_batch_storage import (
    CachePreScan,
    pre_scan_cache,
    stable_cache_id,
    write_failure_json,
)


@dataclass
class _Item:
    work_id: str


def test_stable_cache_id_is_deterministic():
    cid1 = stable_cache_id(
        work_id="w1",
        prompt_sha="p",
        schema_sha="s",
        model_name="m",
        temperature=0.0,
        judge_enabled=True,
        judge_prompt_sha="j",
    )
    cid2 = stable_cache_id(
        work_id="w1",
        prompt_sha="p",
        schema_sha="s",
        model_name="m",
        temperature=0.0,
        judge_enabled=True,
        judge_prompt_sha="j",
    )
    assert cid1 == cid2
    assert isinstance(cid1, str)
    assert len(cid1) == 40  # sha1 hex


@pytest.mark.parametrize(
    "kwargs_change",
    [
        {"work_id": "w2"},
        {"prompt_sha": "p2"},
        {"schema_sha": "s2"},
        {"model_name": "m2"},
        {"temperature": 0.1},
        {"judge_enabled": False},
        {"judge_prompt_sha": "j2"},
    ],
)
def test_stable_cache_id_changes_when_inputs_change(kwargs_change):
    base = dict(
        work_id="w1",
        prompt_sha="p",
        schema_sha="s",
        model_name="m",
        temperature=0.0,
        judge_enabled=True,
        judge_prompt_sha="j",
    )
    cid_base = stable_cache_id(**base)

    changed = dict(base)
    changed.update(kwargs_change)
    cid_changed = stable_cache_id(**changed)

    assert cid_base != cid_changed


def test_pre_scan_cache_counts_and_runnable_indices(tmp_path: Path):
    outputs_dir = tmp_path / "llm_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    items = [_Item("w1"), _Item("w2"), _Item("w3")]

    # precompute cache ids so we can create cache hit files for w1 and w3
    cid_w1 = stable_cache_id(
        work_id="w1",
        prompt_sha="p",
        schema_sha="s",
        model_name="m",
        temperature=0.0,
        judge_enabled=True,
        judge_prompt_sha="j",
    )
    cid_w3 = stable_cache_id(
        work_id="w3",
        prompt_sha="p",
        schema_sha="s",
        model_name="m",
        temperature=0.0,
        judge_enabled=True,
        judge_prompt_sha="j",
    )
    (outputs_dir / f"{cid_w1}.json").write_text("{}", encoding="utf-8")
    (outputs_dir / f"{cid_w3}.json").write_text("{}", encoding="utf-8")

    scan = pre_scan_cache(
        items=items,
        outputs_dir=outputs_dir,
        cache_enabled=True,
        cache_force=False,
        prompt_sha="p",
        schema_sha="s",
        model_name="m",
        temperature=0.0,
        judge_enabled=True,
        judge_prompt_sha="j",
    )

    assert isinstance(scan, CachePreScan)
    assert scan.n_total == 3
    assert scan.n_cache_skips == 2
    assert scan.n_will_run == 1
    assert scan.runnable_indices == [1]  # only w2 should run
    assert scan.cache_id_by_work_id["w1"] == cid_w1
    assert scan.cache_id_by_work_id["w3"] == cid_w3


def test_pre_scan_cache_force_true_runs_everything(tmp_path: Path):
    outputs_dir = tmp_path / "llm_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    items = [_Item("w1"), _Item("w2")]

    # create cache hit for w1
    cid_w1 = stable_cache_id(
        work_id="w1",
        prompt_sha="p",
        schema_sha="s",
        model_name="m",
        temperature=0.0,
        judge_enabled=False,
        judge_prompt_sha="",
    )
    (outputs_dir / f"{cid_w1}.json").write_text("{}", encoding="utf-8")

    scan = pre_scan_cache(
        items=items,
        outputs_dir=outputs_dir,
        cache_enabled=True,
        cache_force=True,  # force bypasses hits
        prompt_sha="p",
        schema_sha="s",
        model_name="m",
        temperature=0.0,
        judge_enabled=False,
        judge_prompt_sha="",
    )

    assert scan.n_cache_skips == 0
    assert scan.n_will_run == 2
    assert scan.runnable_indices == [0, 1]


def test_pre_scan_cache_missing_work_id_raises(tmp_path: Path):
    outputs_dir = tmp_path / "llm_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    class _Bad:
        work_id = ""

    with pytest.raises(ValueError):
        pre_scan_cache(
            items=[_Bad()],  # type: ignore[list-item]
            outputs_dir=outputs_dir,
            cache_enabled=True,
            cache_force=False,
            prompt_sha="p",
            schema_sha="s",
            model_name="m",
            temperature=0.0,
            judge_enabled=False,
            judge_prompt_sha="",
        )


def test_write_failure_json_writes_expected_shape(tmp_path: Path):
    failures_dir = tmp_path / "llm_failures"
    failures_dir.mkdir(parents=True, exist_ok=True)

    p = write_failure_json(
        failures_dir=failures_dir,
        cache_id="abc123",
        attempt=2,
        meta={"work_id": "w1", "attempt_outer": 2},
        err=RuntimeError("boom"),
    )

    assert p.exists()
    assert p.name == "abc123__a2.json"

    obj = json.loads(p.read_text(encoding="utf-8"))
    assert "meta" in obj
    assert obj["meta"]["work_id"] == "w1"
    assert obj["error"]["type"] == "RuntimeError"
    assert obj["error"]["message"] == "boom"
