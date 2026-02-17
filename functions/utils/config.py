# functions/utils/config.py
"""
Config Loader — Universal LLM Batch Generation Framework (Typed YAML Configs)

Intent
- Load + validate universal YAML configuration files:
  - configs/parameters.yaml
  - configs/credentials.yaml
- Return typed configuration objects (Pydantic v2) aligned with the universal framework.

What this module guarantees
- Strict validation: invalid configs fail fast with actionable Pydantic errors.
- Unicode whitespace hardening BEFORE YAML parse: NBSP/BOM/narrow NBSP normalized.
- Minimal filesystem setup via ensure_dirs() for cache / outputs / reports / logs / schema archive.

Forward compatibility
- Accepts older configs that used:
  - llm_schema.path (maps to py_path)
  - cache.artifact_dir (maps to cache.dir)
  - missing context/artifacts/run blocks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from functions.utils.logging import get_logger


# -----------------------------
# Parameter models (Universal)
# -----------------------------

InputFormat = Literal["csv", "tsv", "psv", "xlsx"]
GroupingMode = Literal["group_output", "row_output_with_group_context"]

ContextColumnsMode = Literal["all", "include", "exclude"]
KVOrder = Literal["input_order", "alpha"]

OutputFormat = Literal["psv", "jsonl"]


class RunConfig(BaseModel):
    name: str = "universal_llm_batch_gen_framework"
    timezone: str = "Asia/Tokyo"
    log_level: str = "INFO"
    log_file: Optional[str] = None
    run_id: Optional[str] = None


class InputConfig(BaseModel):
    path: str = "raw_data/input.csv"
    format: InputFormat = "csv"
    encoding: str = "utf-8"
    sheet: Optional[str] = None  # for xlsx only
    required_columns: Optional[list[str]] = None


class GroupingConfig(BaseModel):
    enabled: bool = False
    column: Optional[str] = None
    mode: GroupingMode = "group_output"
    max_rows_per_group: int = 50

    @field_validator("max_rows_per_group")
    @classmethod
    def _validate_max_rows_per_group(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("grouping.max_rows_per_group must be > 0")
        return v


class ContextColumnsConfig(BaseModel):
    mode: ContextColumnsMode = "all"
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class ContextConfig(BaseModel):
    """
    Context construction controls for LLM generation.

    Used by functions/core/context_builder.py
    """
    columns: ContextColumnsConfig = Field(default_factory=ContextColumnsConfig)

    row_template: str = "{__ROW_KV_BLOCK__}"
    auto_kv_block: bool = True
    kv_order: KVOrder = "input_order"

    max_context_chars: int = 12000
    truncate_field_chars: int = 2000

    group_header_template: Optional[str] = None
    group_footer_template: Optional[str] = None

    @field_validator("max_context_chars", "truncate_field_chars")
    @classmethod
    def _validate_nonnegative_int(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError(f"context.{info.field_name} must be >= 0")
        return v


class GenerationPromptConfig(BaseModel):
    path: str = "prompts/generation.yaml"


class JudgePromptConfig(BaseModel):
    enabled: bool = False
    path: str = "prompts/judge.yaml"


class SchemaAutoPyPromptConfig(BaseModel):
    path: str = "prompts/schema_auto_py_generation.yaml"


class SchemaAutoJsonPromptConfig(BaseModel):
    path: str = "prompts/schema_auto_json_summarization.yaml"


class PromptsConfig(BaseModel):
    generation: GenerationPromptConfig = Field(default_factory=GenerationPromptConfig)
    judge: JudgePromptConfig = Field(default_factory=JudgePromptConfig)
    schema_auto_py_generation: SchemaAutoPyPromptConfig = Field(default_factory=SchemaAutoPyPromptConfig)
    schema_auto_json_summarization: SchemaAutoJsonPromptConfig = Field(default_factory=SchemaAutoJsonPromptConfig)


class SchemaConfig(BaseModel):
    """
    Current (preferred):
      - py_path
      - txt_path

    Backward compatibility:
      - path -> py_path
    """
    py_path: str = "schema/llm_schema.py"
    txt_path: str = "schema/llm_schema.txt"
    auto_generate: bool = True
    force_regenerate: bool = False
    archive_dir: str = "archived/"

    # Back-compat input field (optional)
    path: Optional[str] = None

    @model_validator(mode="after")
    def _apply_backcompat(self) -> "SchemaConfig":
        if (not self.py_path or self.py_path == "schema/llm_schema.py") and self.path:
            # If older config provided `path`, use it as py_path
            self.py_path = self.path
        return self


class LLMConfig(BaseModel):
    model_name: str = "gemini-3-flash-preview"
    temperature: float = 1.0
    max_retries: int = 3
    timeout_sec: int = 60
    max_workers: int = 10
    silence_client_lv_logs: bool = True

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError("llm.temperature must be within [0.0, 2.0]")
        return v

    @field_validator("max_retries", "timeout_sec", "max_workers")
    @classmethod
    def _validate_positive_int(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"llm.{info.field_name} must be > 0")
        return v


class CacheConfig(BaseModel):
    """
    Current (preferred):
      - dir

    Backward compatibility:
      - artifact_dir -> dir
    """
    enabled: bool = True
    force: bool = False
    dir: str = "artifacts/cache"
    dump_failures: bool = True
    verbose: int = 0  # NEW: pipeline verbosity (0..10)

    # Back-compat input field (optional)
    artifact_dir: Optional[str] = None

    @field_validator("verbose")
    @classmethod
    def _validate_verbose(cls, v: int) -> int:
        if v < 0 or v > 10:
            raise ValueError("cache.verbose must be within [0, 10]")
        return v

    @model_validator(mode="after")
    def _apply_backcompat(self) -> "CacheConfig":
        if self.artifact_dir and (not self.dir or self.dir == "artifacts/cache"):
            self.dir = self.artifact_dir
        return self


class ArtifactsConfig(BaseModel):
    dir: str = "artifacts"
    outputs_dir: str = "artifacts/outputs"
    reports_dir: str = "artifacts/reports"
    logs_dir: str = "artifacts/logs"


class OutputsConfig(BaseModel):
    formats: list[OutputFormat] = Field(default_factory=lambda: ["psv", "jsonl"])
    psv_path: str = "artifacts/outputs/output.psv"
    jsonl_path: str = "artifacts/outputs/output.jsonl"

    @field_validator("formats")
    @classmethod
    def _validate_formats_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("outputs.formats must contain at least one format (psv/jsonl)")
        return v


class ReportConfig(BaseModel):
    enabled: bool = True
    md_path: str = "artifacts/reports/report.md"
    html_path: str = "artifacts/reports/report.html"

    # NEW (Pipeline 6 — generalized)
    write_html: bool = True
    sample_per_group: int = 2
    include_full_examples: bool = False
    max_reason_examples: int = 5

    # Backward compatibility (old name)
    sample_per_role: Optional[int] = None

    @model_validator(mode="after")
    def _apply_backcompat(self) -> "ReportConfig":
        # If old config used sample_per_role, map it to sample_per_group
        if self.sample_per_role is not None:
            self.sample_per_group = self.sample_per_role
        return self

    @field_validator("sample_per_group", "max_reason_examples")
    @classmethod
    def _validate_nonnegative_int(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError(f"report.{info.field_name} must be >= 0")
        return v


class ParametersConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)

    input: InputConfig = Field(default_factory=InputConfig)
    grouping: GroupingConfig = Field(default_factory=GroupingConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)

    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    llm_schema: SchemaConfig = Field(default_factory=SchemaConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @field_validator("grouping")
    @classmethod
    def _validate_grouping_block(cls, g: GroupingConfig) -> GroupingConfig:
        if g.enabled and (g.column is None or not str(g.column).strip()):
            raise ValueError("grouping.column is required when grouping.enabled=true")
        return g


# -----------------------------
# Credentials models
# -----------------------------

class CredentialsGeminiRequest(BaseModel):
    timeout_seconds: int = 60
    retry_backoff_seconds: int = 2
    max_retry_backoff_seconds: int = 20


class CredentialsGemini(BaseModel):
    api_key_env: str = "GEMINI_API_KEY"
    model_name: Optional[str] = None
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None
    request: CredentialsGeminiRequest = Field(default_factory=CredentialsGeminiRequest)


class CredentialsConfig(BaseModel):
    gemini: CredentialsGemini = Field(default_factory=CredentialsGemini)


# -----------------------------
# YAML helpers
# -----------------------------

_BAD_WHITESPACE = ["\u00A0", "\u2007", "\u202F", "\uFEFF"]


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {str(path)}")

    raw = p.read_text(encoding="utf-8")

    # sanitize BEFORE YAML parse (fix NBSP / BOM / narrow NBSP)
    for ch in _BAD_WHITESPACE:
        raw = raw.replace(ch, " ")

    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/object: {str(path)}")
    return data


def load_parameters(path: str | Path = "configs/parameters.yaml") -> ParametersConfig:
    logger = get_logger(__name__)
    raw = _load_yaml(path)
    try:
        return ParametersConfig.model_validate(raw)
    except ValidationError as e:
        logger.error("Invalid parameters.yaml: %s", e)
        raise


def load_credentials(path: str | Path = "configs/credentials.yaml") -> CredentialsConfig:
    logger = get_logger(__name__)
    raw = _load_yaml(path)
    try:
        return CredentialsConfig.model_validate(raw)
    except ValidationError as e:
        logger.error("Invalid credentials.yaml: %s", e)
        raise


def ensure_dirs(params: ParametersConfig) -> None:
    """
    Ensure configured output directories exist.

    Creates:
    - cache.dir
    - artifacts.outputs_dir / artifacts.reports_dir / artifacts.logs_dir
    - parent dir for outputs.psv_path and outputs.jsonl_path
    - report output directories
    - llm_schema.archive_dir
    - parent dir for llm_schema.py_path and llm_schema.txt_path
    """
    dirs = [
        params.cache.dir,
        params.artifacts.outputs_dir,
        params.artifacts.reports_dir,
        params.artifacts.logs_dir,
        str(Path(params.outputs.psv_path).parent),
        str(Path(params.outputs.jsonl_path).parent),
        str(Path(params.report.md_path).parent),
        str(Path(params.report.html_path).parent),
        params.llm_schema.archive_dir,
        str(Path(params.llm_schema.py_path).parent),
        str(Path(params.llm_schema.txt_path).parent),
    ]

    for d in dirs:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)


__all__ = [
    "ParametersConfig",
    "CredentialsConfig",
    "load_parameters",
    "load_credentials",
    "ensure_dirs",
]
