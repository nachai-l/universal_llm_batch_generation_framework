# functions/llm/runner.py
"""
Gemini Prompt Runner (JSON-only + Schema Validation + Cache)

Intent
- Provide one high-level entrypoint to run a prompt by `prompt_key` with variables,
  call Gemini, and return **validated JSON** as a Pydantic model.
- Make LLM execution predictable and pipeline-friendly by enforcing:
  - JSON extraction (even if the model adds fences or trailing text)
  - schema validation (Pydantic)
  - bounded retries with corrective instructions
  - deterministic, caller-supplied caching via `cache_id`
  - optional failure dumps for debugging

What this module guarantees
- **Return type safety:** `run_prompt_json()` returns a `BaseModel` instance validated
  against the provided `schema_model`.
- **JSON-only enforcement:** model output is parsed to the first JSON object/array;
  markdown/code fences and trailing junk are ignored.
- **Stable caching semantics:**
  - HIT: cached payload exists and validates against schema
  - STALE: cached payload exists but fails schema validation → re-call Gemini
  - MISS: no cache file found
  - FORCE: cache read bypassed via `force=True`
- **Debuggability:** if enabled, invalid model outputs are saved under
  `artifacts/cache/_failures/` so you can inspect failures without re-running.

Dependencies
- Prompt rendering:
  - `functions.llm.prompts.load_prompt_templates()`
  - `functions.llm.prompts.render_prompt()`
- Validation:
  - Pydantic `BaseModel` / `ValidationError`
- Gemini SDK:
  - `client.models.generate_content(...)` (via `google.genai` client instance passed in)

Key functions
- `run_prompt_json(...) -> BaseModel`
  Renders prompt from `configs/prompts.yaml`, runs Gemini, extracts JSON,
  validates the payload, and optionally caches the validated payload.

Supporting utilities (internal)
- `_extract_json(text)`: strip fences and decode the first JSON value
- `_auto cache`: `_try_read_cache()` / `_write_cache()` with filesystem-safe cache ids
- `_write_failure_dump()`: best-effort raw output dump for invalid attempts
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Type, Tuple

from pydantic import BaseModel, ValidationError

from functions.llm.prompts import load_prompt_templates, render_prompt
from functions.utils.logging import get_logger


# Matches ```json ... ``` or ``` ... ```
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)


def _sanitize_cache_id(cache_id: str) -> str:
    """
    Make cache_id filesystem-safe without hashing.
    """
    s = cache_id.strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s[:200]  # keep filename reasonable


def _cache_path(cache_dir: str | Path, cache_id: str) -> Path:
    cid = _sanitize_cache_id(cache_id)
    return Path(cache_dir) / f"{cid}.json"


def _try_read_cache(cache_dir: str | Path, cache_id: Optional[str]) -> Optional[dict]:
    if not cache_id:
        return None
    p = _cache_path(cache_dir, cache_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(cache_dir: str | Path, cache_id: str, payload: dict) -> None:
    p = _cache_path(cache_dir, cache_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_failure_dump(
    cache_dir: str | Path,
    cache_id: Optional[str],
    prompt_key: str,
    attempt: int,
    out_text: str,
) -> None:
    """
    Best-effort: write raw model output for debugging when JSON parsing/validation fails.
    This helps you inspect what the model actually returned without re-running.
    """
    try:
        root = Path(cache_dir) / "_failures"
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cid = _sanitize_cache_id(cache_id or "no_cache_id")
        p = root / f"{cid}__{prompt_key}__a{attempt}__{ts}.txt"
        p.write_text(out_text, encoding="utf-8")
    except Exception:
        # Never fail the pipeline due to debug logging
        return


def _strip_code_fences(text: str) -> str:
    """
    If response is wrapped in markdown code fences, extract inner content.
    If multiple fences exist, prefer the first.
    """
    m = _CODE_FENCE_RE.search(text)
    if m:
        inner = m.group(1)
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
    return text.strip()


def _raw_decode_first_json(text: str) -> Tuple[Any, int]:
    """
    Robustly parse the first JSON value from a string using json.JSONDecoder.raw_decode().

    - Finds the first '{' or '[' and attempts to raw_decode from there.
    - Returns (obj, end_idx) where end_idx is relative to the substring passed to raw_decode.

    Raises ValueError / JSONDecodeError on failure.
    """
    s = text.lstrip("\ufeff \t\r\n")  # handle BOM + whitespace
    # locate first plausible JSON start
    i_obj = s.find("{")
    i_arr = s.find("[")
    if i_obj == -1 and i_arr == -1:
        raise ValueError("No JSON object/array start found in response text")

    start = min([i for i in (i_obj, i_arr) if i != -1])
    s2 = s[start:]

    dec = json.JSONDecoder()
    obj, end = dec.raw_decode(s2)
    return obj, end


def _extract_json(text: str) -> Any:
    """
    Parse JSON from model output.

    Strategy:
    1) Strip code fences if present (```json ... ```).
    2) Try json.loads(full_text).
    3) Fallback: raw_decode the FIRST JSON object/array and ignore trailing text.
       This fixes "Extra data" and most "JSON + commentary" outputs.
    """
    t = _strip_code_fences(text)

    # Fast path: pure JSON
    try:
        return json.loads(t)
    except Exception:
        pass

    # Robust path: decode first JSON and ignore trailing junk
    obj, _end = _raw_decode_first_json(t)
    return obj


def _corrective_prefix(error_msg: str) -> str:
    # Keep short to avoid bloating tokens
    return (
        "CRITICAL: Your previous output was invalid.\n"
        "You MUST output valid JSON only with the exact required schema.\n"
        "Do NOT include markdown, code fences, comments, or extra text.\n"
        f"Error: {error_msg}\n\n"
    )


def _call_gemini_text(
    client: Any,
    model_name: str,
    prompt: str,
    temperature: float,
) -> str:
    """
    Call google.genai client and return response text.
    This function is isolated for test mocking.
    """
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"temperature": temperature},
    )

    # Be defensive across versions
    if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
        return resp.text

    # Fallback: try candidates
    if hasattr(resp, "candidates") and resp.candidates:
        cand0 = resp.candidates[0]
        if hasattr(cand0, "content") and cand0.content:
            return str(cand0.content)

    return str(resp)


def run_prompt_json(
    prompt_key: str,
    variables: Dict[str, Any],
    schema_model: Type[BaseModel],
    client_ctx: Dict[str, Any],
    *,
    prompts_path: str = "configs/prompts.yaml",
    temperature: float = 0.3,
    max_retries: int = 3,
    json_only: bool = True,  # kept for compatibility; extraction always expects JSON
    cache_dir: str | Path = "artifacts/cache",
    cache_id: Optional[str] = None,
    force: bool = False,            # bypass cache read
    write_cache: bool = True,       # allow disabling cache write
    dump_failures: bool = True,     # NEW: write raw bad outputs to artifacts/cache/_failures
) -> BaseModel:
    """
    Render prompt, call Gemini, parse JSON, validate against schema_model, return parsed BaseModel.

    Caching:
    - If cache_id is provided and force=False:
        - try to read artifacts/cache/<cache_id>.json first.
        - if valid, returns cached parsed schema_model (cache=HIT).
        - if exists but invalid for schema, ignore and re-call (cache=STALE).
        - if not exists, cache=MISS.
    - If force=True:
        - skip cache read (cache=FORCE) and always call model.
    - After a successful call, writes validated JSON payload to cache (if write_cache=True).

    Notes:
    - No hashing is used; cache_id must be deterministic and provided by caller.
    """
    logger = get_logger(__name__)

    cache_path: Optional[Path] = _cache_path(cache_dir, cache_id) if cache_id else None

    # 1) cache read (unless forced)
    if cache_id and not force:
        cached = _try_read_cache(cache_dir, cache_id)
        if cached is not None:
            try:
                parsed = schema_model.model_validate(cached)
                logger.debug("LLM cache=HIT | %s | prompt_key=%s", str(cache_path), prompt_key)
                return parsed
            except ValidationError:
                logger.debug("LLM cache=STALE | %s | prompt_key=%s", str(cache_path), prompt_key)
        else:
            logger.debug("LLM cache=MISS | %s | prompt_key=%s", str(cache_path), prompt_key)
    elif cache_id and force:
        logger.debug("LLM cache=FORCE | %s | prompt_key=%s", str(cache_path), prompt_key)

    prompts = load_prompt_templates(prompts_path)
    if prompt_key not in prompts:
        raise KeyError(f"Unknown prompt_key: {prompt_key}")

    template = prompts[prompt_key]
    prompt = render_prompt(template, variables)

    client = client_ctx["client"]
    model_name = client_ctx["model_name"]

    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            out_text = _call_gemini_text(
                client=client,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
            )

            # We always expect JSON payload
            payload = _extract_json(out_text)

            parsed = schema_model.model_validate(payload)

            # cache write
            if cache_id and write_cache:
                _write_cache(cache_dir, cache_id, payload)

            return parsed

        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            last_err = e
            logger.warning("LLM output invalid (attempt %d/%d): %s", attempt, max_retries, str(e))

            if dump_failures:
                _write_failure_dump(
                    cache_dir=cache_dir,
                    cache_id=cache_id,
                    prompt_key=prompt_key,
                    attempt=attempt,
                    out_text=out_text if "out_text" in locals() else "",
                )

            # retry with corrective prefix
            prompt = _corrective_prefix(str(e)) + prompt
            continue

    raise RuntimeError(f"Failed to produce valid JSON after {max_retries} attempts: {last_err}")


__all__ = ["run_prompt_json"]
