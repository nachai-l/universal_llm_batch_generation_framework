# functions/llm/prompts.py
"""
Prompt Templates (YAML) — Loading + Deterministic Rendering

Intent
- Load prompt templates from `configs/prompts.yaml` and render them with runtime variables.
- Keep prompt rendering deterministic and reproducible across runs.
- Harden prompt text against problematic Unicode whitespace (e.g., NBSP) that can silently
  break YAML block scalars or indentation when prompts are edited in tools like Word/Notion.

What this module guarantees
- **Safe prompt loading:** every template loaded from YAML is sanitized to replace common
  non-ASCII whitespace characters with regular spaces and normalize CRLF → LF.
- **Deterministic rendering:** rendering uses `str.format(**variables)` and expects callers to
  inject pre-serialized JSON (preferably with stable key ordering).
- **Clear failures:** missing template variables raise a `KeyError` that explicitly names the
  missing placeholder.

Primary functions
- `load_prompt_templates(path="configs/prompts.yaml") -> Dict[str, str]`
  Loads templates via `functions.utils.config.load_prompts()` and sanitizes each template.

- `render_prompt(template: str, variables: Dict[str, Any]) -> str`
  Renders templates using `str.format`. Raises a clear error for missing variables.

- `build_variables_for_common_enums(params) -> Dict[str, str]`
  Convenience helper to convert common parameter enum lists into stable JSON strings.

Notes & usage conventions
- Prefer injecting structured objects as JSON strings created with stable serialization:
  - `json.dumps(obj, ensure_ascii=False, sort_keys=True)`
  This avoids non-deterministic ordering and reduces diff noise in cached runs.
- Sanitization is intentionally conservative: it only normalizes whitespace known to cause
  YAML/indentation issues; it does not attempt to “fix” prompt content beyond that.

External dependencies
- `functions.utils.config.load_prompts`
- Python stdlib: `json`, `str.format`
"""

from __future__ import annotations

import json
from typing import Any, Dict

from functions.utils.config import load_prompts


# Unicode spaces that commonly break YAML / indentation
_BAD_WHITESPACE = {
    "\u00A0",  # NO-BREAK SPACE
    "\u2007",  # FIGURE SPACE
    "\u202F",  # NARROW NO-BREAK SPACE
    "\uFEFF",  # BOM
}


def _sanitize_prompt_text(text: str) -> str:
    """
    Normalize problematic Unicode whitespace to regular ASCII spaces.
    This prevents YAML block-scalar indentation issues from editors like Word / Notion.
    """
    if not isinstance(text, str):
        return text

    for ch in _BAD_WHITESPACE:
        text = text.replace(ch, " ")

    # Also normalize CRLF just in case
    return text.replace("\r\n", "\n")


def load_prompt_templates(path: str = "configs/prompts.yaml") -> Dict[str, str]:
    """
    Load prompts.yaml via config loader and return prompt templates mapping.
    Sanitizes prompt text to avoid Unicode whitespace YAML failures.
    """
    prompts = load_prompts(path)

    # Sanitize every prompt template defensively
    clean: Dict[str, str] = {}
    for key, template in prompts.items():
        clean[key] = _sanitize_prompt_text(template)

    return clean


def _json_dumps_stable(obj: Any) -> str:
    """
    Stable JSON serializer for prompt injection:
    - sort_keys=True for deterministic ordering
    - ensure_ascii=False for Thai text
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    """
    Render a prompt template using Python str.format(**variables).

    Rules:
    - Deterministic: caller should pass JSON strings or use json.dumps stable helpers.
    - Raises KeyError with a clear message if any variable is missing.
    """
    try:
        return template.format(**variables)
    except KeyError as e:
        missing = str(e).strip("'")
        raise KeyError(f"Missing template variable: {missing}") from e


def build_variables_for_common_enums(params: Any) -> Dict[str, str]:
    """
    Convenience helper: convert enum lists in parameters into JSON strings for prompt rendering.
    Expects params.skills.categories, params.skills.specificity_levels, params.alignment.match_types.
    """
    return {
        "categories_json": _json_dumps_stable(params.skills.categories),
        "specificity_json": _json_dumps_stable(params.skills.specificity_levels),
        "match_types_json": _json_dumps_stable(params.alignment.match_types),
    }


__all__ = [
    "load_prompt_templates",
    "render_prompt",
    "build_variables_for_common_enums",
]
