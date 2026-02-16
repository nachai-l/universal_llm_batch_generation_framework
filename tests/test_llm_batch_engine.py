# tests/test_llm_batch_engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, ConfigDict

from functions.core.llm_batch_engine import (
    GenerationResult,
    generate_with_optional_judge,
)


class _GenOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    foo: str


class _JudgeA(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    feedback: str = ""


class _JudgeB(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    message: str = ""


def test_generate_no_judge_success_first_try(tmp_path: Path):
    calls: List[Dict[str, Any]] = []

    def _fake_runner(*, prompt_path, variables, schema_model, client_ctx, **kwargs):
        calls.append(
            {
                "prompt_path": str(prompt_path),
                "variables": dict(variables),
                "schema_model": schema_model,
                "client_ctx": dict(client_ctx),
                "kwargs": dict(kwargs),
            }
        )
        return _GenOut(foo="ok")

    res = generate_with_optional_judge(
        context="CTX",
        llm_schema_text="{}",
        gen_prompt_path=tmp_path / "gen.yaml",
        judge_prompt_path=None,
        gen_model=_GenOut,
        judge_model=None,
        client_ctx={"client": object(), "model_name": "dummy"},
        temperature=0.0,
        max_retries_outer=3,
        runner_max_retries=2,
        cache_dir="artifacts/cache",
        runner=_fake_runner,
    )

    assert isinstance(res, GenerationResult)
    assert res.status == "ok"
    assert res.used_attempts == 1
    assert res.parsed is not None and res.parsed.foo == "ok"
    assert res.judge is None
    assert res.last_error is None
    assert len(calls) == 1
    assert calls[0]["variables"]["context"] == "CTX"


def test_generate_with_judge_pass_first_try(tmp_path: Path):
    calls: List[str] = []

    def _fake_runner(*, prompt_path, variables, schema_model, client_ctx, **kwargs):
        calls.append(Path(str(prompt_path)).name)
        if str(prompt_path).endswith("judge.yaml"):
            return _JudgeA(passed=True, feedback="")
        return _GenOut(foo="ok")

    res = generate_with_optional_judge(
        context="CTX",
        llm_schema_text="{}",
        gen_prompt_path=tmp_path / "gen.yaml",
        judge_prompt_path=tmp_path / "judge.yaml",
        gen_model=_GenOut,
        judge_model=_JudgeA,
        client_ctx={"client": object(), "model_name": "dummy"},
        temperature=0.0,
        max_retries_outer=3,
        runner_max_retries=2,
        cache_dir="artifacts/cache",
        runner=_fake_runner,
    )

    assert res.status == "ok"
    assert res.used_attempts == 1
    assert res.parsed is not None and res.parsed.foo == "ok"
    assert res.judge is not None
    assert calls == ["gen.yaml", "judge.yaml"]


def test_generate_with_judge_fail_then_pass_appends_feedback(tmp_path: Path):
    attempt = {"n": 0}
    seen_contexts: List[str] = []

    def _fake_runner(*, prompt_path, variables, schema_model, client_ctx, **kwargs):
        name = Path(str(prompt_path)).name
        if name == "gen.yaml":
            seen_contexts.append(str(variables["context"]))
            return _GenOut(foo="ok")

        # judge.yaml
        attempt["n"] += 1
        if attempt["n"] == 1:
            return _JudgeA(passed=False, feedback="please fix JSON")
        return _JudgeA(passed=True, feedback="")

    res = generate_with_optional_judge(
        context="BASE_CTX",
        llm_schema_text="{}",
        gen_prompt_path=tmp_path / "gen.yaml",
        judge_prompt_path=tmp_path / "judge.yaml",
        gen_model=_GenOut,
        judge_model=_JudgeA,
        client_ctx={"client": object(), "model_name": "dummy"},
        temperature=0.0,
        max_retries_outer=3,
        runner_max_retries=2,
        cache_dir="artifacts/cache",
        runner=_fake_runner,
    )

    assert res.status == "ok"
    assert res.used_attempts == 2
    assert len(seen_contexts) == 2
    assert seen_contexts[0] == "BASE_CTX"
    assert "JUDGE FEEDBACK" in seen_contexts[1]
    assert "please fix JSON" in seen_contexts[1]


def test_generate_with_judge_fail_exhausts_retries(tmp_path: Path):
    def _fake_runner(*, prompt_path, variables, schema_model, client_ctx, **kwargs):
        if str(prompt_path).endswith("judge.yaml"):
            return _JudgeA(passed=False, feedback="nope")
        return _GenOut(foo="ok")

    res = generate_with_optional_judge(
        context="CTX",
        llm_schema_text="{}",
        gen_prompt_path=tmp_path / "gen.yaml",
        judge_prompt_path=tmp_path / "judge.yaml",
        gen_model=_GenOut,
        judge_model=_JudgeA,
        client_ctx={"client": object(), "model_name": "dummy"},
        temperature=0.0,
        max_retries_outer=2,
        runner_max_retries=2,
        cache_dir="artifacts/cache",
        runner=_fake_runner,
    )

    assert res.status == "fail"
    assert res.used_attempts == 2
    assert res.last_error is not None
    assert "Judge failed after retries" in res.last_error


def test_judge_field_variants_ok_message(tmp_path: Path):
    def _fake_runner(*, prompt_path, variables, schema_model, client_ctx, **kwargs):
        if str(prompt_path).endswith("judge.yaml"):
            return _JudgeB(ok=True, message="fine")
        return _GenOut(foo="ok")

    res = generate_with_optional_judge(
        context="CTX",
        llm_schema_text="{}",
        gen_prompt_path=tmp_path / "gen.yaml",
        judge_prompt_path=tmp_path / "judge.yaml",
        gen_model=_GenOut,
        judge_model=_JudgeB,
        client_ctx={"client": object(), "model_name": "dummy"},
        temperature=0.0,
        max_retries_outer=1,
        runner_max_retries=2,
        cache_dir="artifacts/cache",
        runner=_fake_runner,
    )
    assert res.status == "ok"
    assert res.used_attempts == 1


def test_runner_kwargs_do_not_enable_cache(tmp_path: Path):
    observed: Dict[str, Any] = {}

    def _fake_runner(*, prompt_path, variables, schema_model, client_ctx, **kwargs):
        observed.update(kwargs)
        return _GenOut(foo="ok")

    _ = generate_with_optional_judge(
        context="CTX",
        llm_schema_text="{}",
        gen_prompt_path=tmp_path / "gen.yaml",
        judge_prompt_path=None,
        gen_model=_GenOut,
        judge_model=None,
        client_ctx={"client": object(), "model_name": "dummy"},
        temperature=0.0,
        max_retries_outer=1,
        runner_max_retries=2,
        cache_dir="artifacts/cache",
        runner=_fake_runner,
    )

    # We must NOT ask runner to write cache for pipeline4 candidates
    assert observed.get("cache_id", "MISSING") is None
    assert observed.get("write_cache", "MISSING") is False
    assert observed.get("force", "MISSING") is True
