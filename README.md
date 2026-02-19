# Universal LLM Batch Generation Framework

> A deterministic, schema-driven, judge-gated LLM batch processing framework.
> End-to-end verified (Pipelines 0–6) · Strict schema validation · Artifact-level caching · Reproducible exports · 236+ tests passing

A **universal, task-agnostic LLM batch generation framework** that ingests a tabular input file, builds deterministic work units, runs LLM generation with strict runtime schema validation, optionally applies a judge-gated acceptance step, and exports analysis-ready outputs with a human-readable audit report.

This framework is built to eliminate common LLM batch-processing risks:

* Non-deterministic reruns
* Silent schema drift
* Invalid outputs silently passing downstream
* Untraceable failures
* Cache corruption across task evolution

It enforces structure, traceability, and reproducibility at every stage.

---

# What This Framework Does

The framework performs the following deterministic execution pipeline:

1. Ingest a single structured tabular file (`csv / psv / tsv / xlsx`)
2. Normalize and snapshot the input deterministically
3. Build stable WorkItems (row-wise or grouped)
4. Generate structured outputs via LLM
5. Validate outputs strictly against a runtime Pydantic v2 schema
6. Optionally judge the outputs and retry if necessary
7. Export canonical JSONL + thin PSV tables
8. Produce an audit report (Markdown + HTML)

### Core Guarantees

* **Strict Schema Enforcement** — Pydantic v2 with `extra="forbid"`
* **Deterministic Hashing** — stable `work_id`, `group_context_id`, `cache_id`
* **Cache Discipline** — identical inputs never re-spend tokens
* **Artifact Transparency** — every stage writes inspectable files
* **Safe Retry Loop** — corrective prefix injection on validation failure
* **Forward-Compatible Evolution** — schema or prompt changes produce new cache keys

---

# Design Philosophy

This framework is built on the principle that:

> LLM generation must behave like a deterministic data pipeline, not an interactive chat tool.

To achieve this:

* Pipelines are orchestration-only.
* Business logic lives in `functions/core/`.
* Artifacts are the source of truth.
* Schema is the contract.
* Judge never mutates content.
* Cache keys include semantic configuration components.

The runtime schema (`llm_schema.py`) is the authoritative contract.
The prompt schema (`llm_schema.txt`) is derived from it and must never diverge.

---

# Project Structure

```
.
├── configs/
│   └── parameters.yaml            # All pipeline parameters (typed, validated)
│
├── prompts/                        # YAML prompt templates
│   ├── schema_auto_py_generation.yaml
│   ├── generation.yaml
│   └── judge.yaml
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

### Key Directories Explained

* `functions/core/` — deterministic domain logic (fully unit-testable)
* `functions/batch/` — thin pipeline entrypoints
* `artifacts/cache/` — validated LLM outputs + manifests + failure dumps
* `artifacts/outputs/` — canonical user-facing exports
* `archived/` — timestamped schema history for evolution traceability

---

# Pipeline Overview

```
[0] schema_ensure
[1] schema_txt_ensure
[2] ingest_input
[3] build_requests
[4] llm_generate
[5] export_outputs
[6] write_report
```

Each pipeline:

* Is independently runnable
* Writes a manifest
* Does not silently overwrite artifacts
* Is deterministic given unchanged inputs

---

# Detailed Pipeline Behavior

## Pipeline 0 — Schema Ensure

Ensures a valid, importable Pydantic schema exists.

If auto-generate enabled:

* Archives previous schema
* Generates new one via LLM
* Injects `ConfigDict(extra="forbid")`
* Validates importability and BaseModel presence

This prevents schema drift and silent validation gaps.

---

## Pipeline 1 — Schema Text Ensure

Generates JSON-schema text from runtime schema.

This ensures:

* Prompt contract matches runtime validator
* No hand-maintained duplication
* Strict `additionalProperties: false`

---

## Pipeline 2 — Input Ingestion

* All values treated as strings
* Null tokens normalized
* Whitespace normalized
* Stable snapshot written to `pipeline2_input.json`

This snapshot guarantees reproducibility across reruns.

---

## Pipeline 3 — WorkItem Construction

Supports three modes:

| Mode                          | Description                            |
| ----------------------------- | -------------------------------------- |
| row                           | 1 row → 1 LLM call                     |
| group_output                  | 1 group → 1 LLM call                   |
| row_output_with_group_context | 1 group context → multiple row outputs |

Group context is hashed and de-duplicated.

Stable IDs ensure:

* Identical runs → identical `work_id`
* Context changes → new IDs

---

## Pipeline 4 — LLM Generation

For each WorkItem:

1. Compute deterministic `cache_id`
2. Pre-scan cache
3. Generate output
4. Extract JSON
5. Validate schema
6. Optional judge validation
7. Persist artifact
8. Write manifest

Cache key includes:

```
work_id
prompt_sha
schema_sha
model_name
temperature
judge_enabled
judge_prompt_sha
```

Any semantic change automatically invalidates cache safely.

---

## Pipeline 5 — Export

Outputs:

* `output.jsonl` — full canonical record
* `output.psv` — thin export (spreadsheet-friendly)

PSV characteristics:

* Deterministic column order
* Escaped newline, tab, pipe
* No silent overwrites on field collisions

---

## Pipeline 6 — Reporting

Generates:

* Run statistics
* Group distributions
* Judge pass/fail summary
* Failure diagnostics
* Spot-check samples

Designed for audit and quality monitoring.

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
  model_name: gemini-3-flash-preview
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

# 🔄 How to Prepare a New Use Case

The framework is task-agnostic.
To introduce a new workflow (generation, translation, extraction, scoring, evaluation):

---

## Step 1 — Prepare Input File

Update:

```
raw_data/input.csv
```

Guidelines:

* Include only required columns.
* Remove judge/meta fields if chaining.
* All values treated as strings.
* Ensure column names exactly match configuration.

If chaining from previous output:

* Use `output.psv` or `output.jsonl`
* Remove heavy JSON columns unless required
* Validate grouping column presence

---

## Step 2 — Update parameters.yaml

Key areas to verify:

### Input

```yaml
input:
  path:
  format:
```

### Grouping

```yaml
grouping:
  enabled:
  column:
  mode:
```

### LLM

```yaml
llm:
  model_name:
  temperature:
```

### Cache

If task semantics changed:

```yaml
cache:
  force: true
```

or clear cache manually.

---

## Step 3 — Update generation.yaml

Requirements:

* Must inject `{llm_schema}`
* Must enforce JSON-only output
* Must match schema exactly
* Must not echo schema
* Must not include markdown

Maintain HARD OUTPUT CONTRACT block.

---

## Step 4 — Update judge.yaml (If Enabled)

Judge must:

* Match grouping mode
* Match schema
* Return EXACT shape:

```json
{
  "verdict": "PASS",
  "score": 85,
  "reasons": []
}
```

Disable judge for pure transformations if evaluation unnecessary.

---

## Step 5 — Update Schema

### Auto-Generate (Recommended)

```yaml
llm_schema:
  auto_generate: true
  force_regenerate: true
```

Run Pipeline 0.

### Manual Update

If editing manually:

* Must use Pydantic v2
* Must include `ConfigDict(extra="forbid")`
* Must match generation.yaml exactly

Skip Pipeline 0 only if schema is already correct.

---

## Step 6 — Run Pipelines

Full run:

```bash
python scripts/run_pipeline_0_force.py
...
python scripts/run_pipeline_6_force.py
```

You may start from later pipeline if earlier stages unchanged.

---

## Step 7 — Validate Outputs

Outputs:

```
artifacts/outputs/output.jsonl
artifacts/outputs/output.psv
```

Reports:

```
artifacts/reports/report.md
artifacts/reports/report.html
```

Review:

* Schema correctness
* Judge performance
* Field completeness
* Unexpected truncations
* Group coverage

---

# 🔗 Chaining Workflows

Example:

1. Generate interview questions
2. Export output
3. Feed into translation workflow
4. Judge translation quality
5. Aggregate results

Best Practices:

* Remove unnecessary JSON columns
* Validate schema before each new run
* Clear cache when semantic meaning changes
* Maintain versioned prompts per use case

---

# When Must Schema Be Regenerated?

Regenerate when:

* Fields added or removed
* Field names changed
* Structure changed
* Nested model changed
* Grouping logic affects output shape
* Task type changed (generation → translation)

Do NOT regenerate when:

* Only input rows changed
* Only wording refined
* Only judge logic changed

---

# Determinism Guarantees

If unchanged:

* Input file
* parameters.yaml
* schema
* prompts
* model name
* temperature

Then guaranteed identical:

* work_id
* group_context_id
* cache_id
* JSONL
* PSV
* manifests

Warm cache reruns are byte-identical and zero-cost.

---

# Requirements

* Python 3.12+
* pandas
* pydantic >= 2
* PyYAML
* google-genai

```bash
pip install -r requirements.txt
```

---

# Final Note

This is not a script.
It is a deterministic LLM batch execution engine.

Swap input.  
Swap schema.  
Swap prompt.  

The engine remains stable.

Designed for:

* Reproducible generation
* Auditable workflows
* Safe iteration
* Scalable evaluation
* Structured LLM experimentation
