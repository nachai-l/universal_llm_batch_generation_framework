# functions/core/llm_batch_engine.py
"""
LLM Batch Engine — Generation + Optional Judge (Outer Retry Loop)

Intent
- Provide the core "attempt loop" used by Pipeline 4:
  - run generation prompt
  - optionally run judge prompt
  - if judge fails, append judge feedback to context deterministically and retry
- Keep Pipeline 4 thin and focused on orchestration (I/O + caching + logging).

Notes
- This module intentionally disables runner-level caching for candidates:
  - cache_id=None, write_cache=False, force=True
  because judge-failed candidates must not be treated as final cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type


RunnerFn = Callable[..., Any]


@dataclass(frozen=True)
class GenerationResult:
    status: str  # "ok" | "fail"
    parsed: Optional[Any]
    judge: Optional[Any]
    used_attempts: int
    final_context: str
    last_error: Optional[str]


def _judge_pass_and_feedback(judge_obj: Any) -> Tuple[Optional[bool], str]:
    """
    Interpret judge output flexibly (supports multiple field names).
    Returns:
      (passed_or_none, feedback_text)
    """
    passed: Optional[bool] = None
    feedback: str = ""

    # pass field variants
    for key in ("pass", "passed", "is_pass", "ok", "success"):
        if hasattr(judge_obj, key):
            v = getattr(judge_obj, key)
            if isinstance(v, bool):
                passed = v
                break

    # feedback field variants
    for key in ("feedback", "reason", "message", "detail", "details"):
        if hasattr(judge_obj, key):
            v = getattr(judge_obj, key)
            if isinstance(v, str) and v.strip():
                feedback = v.strip()
                break

    return passed, feedback


def _append_judge_feedback(base_context: str, feedback: str) -> str:
    fb = feedback or "(Judge failed without feedback.)"
    return base_context + "\n\nJUDGE FEEDBACK (fix your output):\n" + fb + "\n"


def generate_with_optional_judge(
    *,
    context: str,
    llm_schema_text: str,
    gen_prompt_path: Any,
    judge_prompt_path: Optional[Any],
    gen_model: Type[Any],
    judge_model: Optional[Type[Any]],
    client_ctx: dict,
    temperature: float,
    max_retries_outer: int,
    runner_max_retries: int,
    cache_dir: str,
    runner: RunnerFn,
) -> GenerationResult:
    """
    Run generation (+ optional judge) with an outer retry loop.

    Args:
      max_retries_outer:
        number of outer attempts when judge fails or any exception occurs.
      runner_max_retries:
        number of inner retries inside runner for JSON validity / schema compliance.

    Returns:
      GenerationResult(status="ok") on success, else status="fail".
    """
    if max_retries_outer <= 0:
        raise ValueError("max_retries_outer must be > 0")

    judge_enabled = bool(judge_prompt_path is not None)

    augmented_context = context
    last_error: Optional[str] = None

    for attempt in range(1, max_retries_outer + 1):
        try:
            # --- generation ---
            gen_vars = {"context": augmented_context, "llm_schema": llm_schema_text}
            gen_obj = runner(
                prompt_path=gen_prompt_path,
                variables=gen_vars,
                schema_model=gen_model,
                client_ctx=client_ctx,
                temperature=temperature,
                max_retries=int(runner_max_retries),
                cache_dir=str(cache_dir),
                cache_id=None,  # IMPORTANT: no runner caching for pipeline4 candidates
                force=True,
                write_cache=False,
                dump_failures=False,
            )

            if not judge_enabled:
                return GenerationResult(
                    status="ok",
                    parsed=gen_obj,
                    judge=None,
                    used_attempts=attempt,
                    final_context=augmented_context,
                    last_error=None,
                )

            # --- judge ---
            if judge_model is None:
                raise RuntimeError("judge_prompt_path provided but judge_model is None")

            judge_vars = {
                "context": augmented_context,
                "llm_schema": llm_schema_text,
                "output_json": json.dumps(gen_obj.model_dump(), ensure_ascii=False, sort_keys=True),
            }
            j_obj = runner(
                prompt_path=judge_prompt_path,
                variables=judge_vars,
                schema_model=judge_model,
                client_ctx=client_ctx,
                temperature=temperature,
                max_retries=int(runner_max_retries),
                cache_dir=str(cache_dir),
                cache_id=None,
                force=True,
                write_cache=False,
                dump_failures=False,
            )

            passed, feedback = _judge_pass_and_feedback(j_obj)
            is_pass = True if passed is None else bool(passed)

            if is_pass:
                return GenerationResult(
                    status="ok",
                    parsed=gen_obj,
                    judge=j_obj,
                    used_attempts=attempt,
                    final_context=augmented_context,
                    last_error=None,
                )

            # judge failed -> retry with feedback appended (deterministic)
            augmented_context = _append_judge_feedback(context, feedback)
            last_error = f"Judge failed: {feedback or '(no feedback)'}"

            if attempt == max_retries_outer:
                last_error = f"Judge failed after retries: {feedback or '(no feedback)'}"
                break

        except Exception as e:
            last_error = str(e)
            if attempt == max_retries_outer:
                break

    return GenerationResult(
        status="fail",
        parsed=None,
        judge=None,
        used_attempts=max_retries_outer,
        final_context=augmented_context,
        last_error=last_error,
    )


__all__ = [
    "GenerationResult",
    "generate_with_optional_judge",
]
