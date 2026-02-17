# Universal LLM Batch Generation Framework

> A deterministic, schema-driven, judge-gated LLM batch processing framework.  
> End-to-end verified (Pipelines 0–6) · Strict schema validation · Artifact-level caching · Reproducible exports · 236 tests passing

A **universal, task-agnostic LLM batch generation framework** that ingests a tabular input file, builds deterministic work units, runs LLM generation with strict schema validation, optionally applies a judge-gated acceptance step, and exports analysis-ready outputs with a human-readable report.

---

## What This Framework Does

- Ingests **one** tabular file (`csv / psv / tsv / xlsx`)
- Builds deterministic **WorkItems** (row-wise or grouped)
- Runs LLM generation with **strict schema validation** (Pydantic v2, `extra="forbid"`)
- Optionally runs a **judge** prompt with automatic retry and feedback injection
- Exports results to `jsonl` (full canonical) + `psv` (thin, spreadsheet-friendly)
- Writes a **report** (`md` + `html`) summarising run health and quality

Designed for:

- **Reproducibility** — hashing, stable ordering, stable serialisation
- **Auditability** — artifact-first, manifest-first, nothing implicit
- **Scalability** — parallel execution, cache hits, group-context de-duplication
- **Forward-compatibility** — schema and prompt evolution without corrupting prior runs

---

## Project Structure

```
.
├── configs/
│   └── parameters.yaml            # All pipeline parameters (typed, validated)
│
├── prompts/                        # YAML prompt templates
│   ├── schema_auto_py_generation.yaml
│   ├── generation_prompt.yaml
│   └── judge_prompt.yaml
│
├── schema/
│   ├── llm_schema.py              # Runtime validation schema (Pydantic v2)
│   └── llm_schema.txt             # Prompt-ready JSON schema (injected into LLM prompts)
│
├── functions/
│   ├── batch/                     # Pipeline entrypoints (thin orchestration only)
│   │   ├── pipeline_0_schema_ensure.py
│   │   ├── pipeline_1_schema_txt_ensure.py
│   │   ├── pipeline_2_ingest_input.py
│   │   ├── pipeline_3_build_requests.py
│   │   ├── pipeline_4_llm_generate.py
│   │   ├── pipeline_5_export_outputs.py
│   │   └── pipeline_6_write_report.py
│   ├── core/                      # Deterministic domain logic
│   │   ├── context_builder.py
│   │   ├── export_outputs.py      # Pipeline 5 core (zero IO, fully testable)
│   │   └── ingestions.py
│   ├── llm/                       # Prompt loading, rendering, LLM runner
│   ├── io/                        # Deterministic readers and writers
│   └── utils/                     # Config loading, logging, hashing
│
├── artifacts/
│   ├── cache/                     # Machine artifacts + manifests + LLM output cache
│   │   ├── pipeline2_input.json
│   │   ├── pipeline3_work_items.json
│   │   ├── pipeline4_manifest.json
│   │   ├── pipeline5_manifest.json
│   │   ├── pipeline6_manifest.json
│   │   ├── llm_outputs/           # Validated success artifacts (1 per WorkItem)
│   │   └── llm_failures/          # Failure dumps (attempt-scoped, never overwritten)
│   ├── outputs/
│   │   ├── output.jsonl           # Full canonical records
│   │   └── output.psv             # Thin PSV table
│   └── reports/
│       ├── report.md
│       └── report.html
│
├── scripts/
│   └── run_pipeline_X_force.py    # Reproducible per-pipeline execution scripts
│
├── tests/                         # pytest suite
├── raw_data/                      # Source input files (git-ignored)
└── archived/                      # Timestamped schema archives
```

---

## Pipeline Overview

```
[0] schema_ensure      → schema/llm_schema.py
[1] schema_txt_ensure  → schema/llm_schema.txt
[2] ingest_input       → artifacts/cache/pipeline2_input.json
[3] build_requests     → artifacts/cache/pipeline3_work_items.json
[4] llm_generate       → artifacts/cache/llm_outputs/* + pipeline4_manifest.json
[5] export_outputs     → artifacts/outputs/output.jsonl + output.psv
[6] write_report       → artifacts/reports/report.md + report.html
```

Each pipeline is independently runnable. Each writes a manifest. No pipeline contains business logic — all logic lives in `functions/core/`.

---

## Pipelines in Detail

### Pipeline 0 — Schema (Python) Ensure

Ensures a valid, importable **Pydantic v2 schema module** (`schema/llm_schema.py`) exists before any generation runs.

- If schema exists and `force_regenerate: false` → no-op
- If missing or `force_regenerate: true`:
  - Archives existing schema (timestamped)
  - Generates schema via LLM prompt
  - Post-processes code: injects `ConfigDict(extra="forbid")` into all `BaseModel` subclasses
  - Validates: syntactically valid Python, importable, contains at least one `BaseModel`

### Pipeline 1 — Schema (Text) Ensure

Converts the Python schema into **prompt-injectable JSON schema text** (`schema/llm_schema.txt`).

- Derived from the Pydantic v2 model via introspection (never hand-written)
- Enforces `additionalProperties: false`
- Guarantees prompt contract matches runtime validation

### Pipeline 2 — Input Ingestion

Reads the raw input table and writes a **deterministic JSON snapshot** (`pipeline2_input.json`).

- Supports: `csv`, `tsv`, `psv`, `xlsx`
- All values read as strings (no implicit type coercion)
- Null tokens normalised: `None`, `NaN`, `n/a`, `NULL`, `[None]` → `""`
- Internal whitespace and newlines collapsed deterministically

### Pipeline 3 — Build WorkItems

Converts the ingested snapshot into **deterministic WorkItems** — the unit of LLM work.

**Supported modes:**

| Mode | Description |
|------|-------------|
| Row-wise | 1 input row → 1 WorkItem |
| Group output | 1 group → 1 WorkItem (LLM sees all group rows) |
| Row output with group context | Group context built once; each row → 1 WorkItem referencing shared context |

**Group context de-duplication:** unique group contexts stored once, referenced by stable `group_context_id` (SHA1 of context content). Eliminates artifact bloat at scale.

**Stable IDs:** `work_id` and `group_context_id` are deterministic across repeated runs for the same data and config.

### Pipeline 4 — LLM Generation (+ Optional Judge)

Executes LLM generation per WorkItem with full validation and optional judge-gating.

**Workflow:**
1. Compute deterministic `cache_id` per WorkItem
2. Cache pre-scan — skip if success artifact exists and `cache.force: false`
3. Run generation prompt
4. Validate strictly against runtime schema (Pydantic v2, `extra="forbid"`)
5. Optional judge: if verdict fails, inject feedback and retry; only judge-approved outputs are persisted
6. Write success artifact or attempt-scoped failure dump
7. Write `pipeline4_manifest.json`

**Parallel execution:** ThreadPoolExecutor (`llm.max_workers`). Determinism preserved by stable ordering and stable cache IDs.

**Cache key** (SHA1 over):
```
work_id + prompt_sha + schema_sha + model_name + temperature + judge_enabled + judge_prompt_sha
```

### Pipeline 5 — Export Outputs

Exports validated artifacts to user-facing formats.

**Core logic** is in `functions/core/export_outputs.py` (zero file IO, fully testable). The pipeline entrypoint is thin orchestration only.

**Outputs:**
- `output.jsonl` — full canonical records, always complete, never lossy
- `output.psv` — thin table, heavy columns excluded by default

**Thin PSV (default):**
Columns excluded from PSV by default (always present in JSONL):

- `group_rows_json`, `group_context_id`, `group_context`, `group_context_meta_json`, `questions_json`

**PSV column ordering (deterministic):**
1. Input columns (source order)
2. Group helper columns (if enabled)
3. Parsed fields (alphabetical)
4. Judge + meta columns (fixed order)

**PSV escape contract:**

| Character | Escaped as |
|-----------|------------|
| Newline | `\n` |
| Tab | `\t` |
| Pipe | `\|` |
| `None` | `""` |

Consumers must unescape these sequences when reading cell values.

**Parsed field collision resolution:** if a parsed field name collides semantically with an input column header (e.g. parsed `question_id` vs input `Question ID`), the parsed field is auto-prefixed with `parsed_` to preserve both without silent overwrite.

### Pipeline 6 — Report Generation

Generates a human-readable run report (`report.md` + `report.html`) covering:

- Run metadata, config, and artifact paths
- Volume stats and distributions (by group, question type, etc.)
- Judge performance: pass/fail counts, score distribution, top reasons
- Data quality checks: missing inputs, missing meta, invalid judge records
- Spot-check samples per group

---

## Configuration

All parameters live in `configs/parameters.yaml` (typed and validated via Pydantic v2).

Key config blocks:

```yaml
input:
  path: raw_data/input.csv
  format: csv                      # csv | tsv | psv | xlsx

grouping:
  enabled: true
  column: "Role Track Example Name"
  mode: row_output_with_group_context   # row_output_with_group_context | group_output
  max_rows_per_group: 50

llm:
  model: gemini-3-flash-preview
  temperature: 1.0
  max_workers: 10
  retries: 5

cache:
  enabled: true
  force: false                     # true = re-run and overwrite success artifacts

judge:
  enabled: true

outputs:
  jsonl_path: artifacts/outputs/output.jsonl
  psv_path: artifacts/outputs/output.psv
  # thin PSV by default
```

---

## Why This Framework Exists

Most LLM batch scripts are:

- Non-deterministic across runs
- Lacking schema enforcement (outputs accepted blindly)
- Without cache discipline (re-run = re-spend)
- Impossible to audit (no artifacts, no manifests)
- Unsafe to rerun (overwrites, data loss)

This framework solves that by making every concern explicit:

| Problem | Solution |
|---------|----------|
| Non-deterministic outputs | Stable hashing, ordering, and serialisation throughout |
| No schema enforcement | Pydantic v2 `extra="forbid"` — invalid outputs never become artifacts |
| No cache discipline | SHA1 cache keys — identical inputs always hit cache |
| No auditability | Every stage writes a manifest; artifacts are never implicitly deleted |
| Judge contamination | Judge runs post-validation; only approved outputs are persisted |
| Schema drift over time | Prompt schema derived from runtime schema — always in sync |

---

## Who This Is For

This framework is suitable for any task that maps tabular input rows to structured LLM-generated outputs:

- Interview question generation (per role / seniority / competency)
- Structured extraction from documents or job descriptions
- Educational content generation (CLO/skill alignment)
- Batch evaluation and scoring workflows
- Research dataset generation
- LLM output benchmarking
- Any task requiring reproducible, auditable batch generation at scale

---

## Design Principles

1. **Pipelines are orchestration-only.** No business logic in pipeline entrypoints.
2. **Artifacts are the source of truth.** Every stage writes inspectable output. Nothing is implicit.
3. **Determinism everywhere.** Stable ordering, serialisation, hashing, and IDs.
4. **Schema is the contract.** Runtime schema is authoritative; prompt schema is derived from it.
5. **LLM output never bypasses validation.** Strict Pydantic v2, `extra="forbid"`.
6. **Judge does not poison cache.** Only judge-approved outputs are persisted as final.
7. **Forward-compatibility.** Schema/prompt changes produce new cache IDs; old runs are never corrupted.
8. **Manifest-first traceability.** Every pipeline writes a manifest with counts, modes, and anomalies.

---

## Running the Framework

Run all pipelines end-to-end (force re-execution):

```bash
python scripts/run_pipeline_0_force.py
python scripts/run_pipeline_1_force.py
python scripts/run_pipeline_2_force.py
python scripts/run_pipeline_3_force.py
python scripts/run_pipeline_4_force.py
python scripts/run_pipeline_5_force.py
python scripts/run_pipeline_6_force.py
```

Clear LLM cache between runs:

```bash
python scripts/clear_llm_cache.py
python scripts/clear_archived.py
```

Run tests:

```bash
pytest -q
```

---

## Determinism Guarantees

If the following are unchanged between runs:

- Input file content
- `configs/parameters.yaml`
- `schema/llm_schema.py`
- Prompt files
- Model name and temperature

Then the following are guaranteed identical:

- All `work_id` and `group_context_id` values
- All `cache_id` values (same inputs always hit cache)
- Export column ordering in PSV
- JSON serialisation order
- Manifest counts

This means reruns with warm cache are always zero-cost and produce byte-identical outputs.

---

## Status

**Framework v1 — Operational ✅**

| Item | State |
|------|-------|
| Pipelines 0–6 | Complete, `rc=0` |
| Strict schema validation | Enforced (`extra="forbid"`) |
| Judge gating | Working (auto-retry with feedback) |
| Artifact-level cache | Stable |
| Thin PSV export | Verified (50 rows × 22 cols) |
| JSONL canonical export | Verified (50 rows, full records) |
| Test suite | 236 tests passing, 0 failures |
| Real execution | End-to-end verified |

---

## Requirements

- Python 3.12+
- pandas
- pydantic >= 2.0
- PyYAML
- google-genai (Gemini SDK)

```bash
pip install -r requirements.txt
```

---

## Notes

- Raw input data and credentials are excluded from version control.
- The framework is fully dataset-agnostic — swap in any tabular input and prompt.
- Designed for **reproducible batch generation, auditing, and analysis**.