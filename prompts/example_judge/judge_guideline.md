# Prompt Guideline: `judge.yaml`

**File:** `prompts/judge.yaml`
**Role in framework:** Post-generation evaluation gate (Pipeline 4)
**Depends on:**

* `generation.yaml`
* `llm_schema.py`
* Pipeline 3 WorkItem context

---

# What This Prompt Does

`judge.yaml` is the **formal evaluation layer** of the framework.

For each generation attempt:

1. The framework injects:

   * Schema
   * Original input context
   * Generated output JSON
2. Sends the rendered judge prompt to the LLM
3. Parses the structured `JudgeResult`
4. If `PASS` → generation artifact is persisted
5. If `FAIL` → judge reasons are injected into the next generation attempt

The judge:

* Does **not** repair content
* Does **not** regenerate content
* Does **not** modify output
* Only evaluates and returns a structured verdict

This prompt is a **gate**, not a helper.

---

# Prompt Structure

Each `judge.yaml` must contain:

| Block     | Purpose                                      |
| --------- | -------------------------------------------- |
| `name`    | Human-readable identifier                    |
| `version` | Manual version tracking                      |
| `purpose` | Short description of evaluation scope        |
| `system`  | Hard evaluator contract + scoring discipline |
| `user`    | Evaluation criteria + injected artifacts     |

---

## Separation of Responsibilities

* `system` → how to evaluate + how to output verdict
* `user` → what to evaluate + evaluation criteria

Never mix them.

Mixing scoring instructions with criteria makes tuning unstable.

---

# Runtime Placeholders (Exact String Match Required)

The framework injects:

| Placeholder     | Source                  | Purpose                               |
| --------------- | ----------------------- | ------------------------------------- |
| `{llm_schema}`  | `schema/llm_schema.txt` | Schema used for generation validation |
| `{context}`     | Pipeline 3              | Original input context                |
| `{output_json}` | Pipeline 4              | Generated output being judged         |

All three **must remain present and unmodified**.

Renaming or removing any placeholder breaks Pipeline 4.

---

# Required Ordering in `user` Block

Recommended structure:

1. Evaluation criteria
2. `{llm_schema}`
3. `{context}`
4. `{output_json}`
5. Final reminder

Keep schema + context + output near the end so the model evaluates against them immediately before responding.

---

# Hard Evaluator Contract (System Block)

This must appear at the top of the `system` block.

```yaml
system: |
  HARD RULES
  - Return VALID JSON only.
  - No markdown.
  - No commentary.
  - No code fences.
  - Do NOT regenerate content.
  - Do NOT repair content.
  - Do NOT modify output.
  - Only evaluate the provided output_json.
```

The "Only evaluate" rule is critical.

Without it, the model attempts to fix content instead of failing it.

---

# Required Judge Output Structure

Judge must return exactly:

```json
{
  "verdict": "PASS" or "FAIL",
  "score": integer 0-100,
  "reasons": ["diagnostic reason 1", "..."]
}
```

All fields required. Always present.

### Field Roles

| Field     | Used By                  | Requirement                          |
| --------- | ------------------------ | ------------------------------------ |
| `verdict` | Retry logic              | Must be EXACTLY `"PASS"` or `"FAIL"` |
| `score`   | Manifest + report        | Integer 0–100                        |
| `reasons` | Retry feedback injection | 1–5 specific diagnostic strings      |

If output shape is incorrect, Pipeline 4 validation fails and retries.

---

# Evaluation Criteria Structure

All criteria must be:

* Numbered
* Independently evaluable
* Explicit about failure conditions
* Explicit about score ceilings
* Clearly marked CRITICAL or non-critical

---

## Criterion 0 — COMPLETENESS (Group Output Mode Only)

Always first if using `group_output`.

```yaml
0) COMPLETENESS (CRITICAL)
Step A: Extract expected identifiers from INPUT CONTEXT.
- Collect values appearing after exact token: "[Identifier Label]:"
- Treat identifiers as strings.

Step B: Extract produced identifiers from output_json.

Requirements:
- Produced identifiers must match expected identifiers exactly.
- No missing.
- No extras.
- No duplicates.

If mismatch → verdict=FAIL and score ≤ 40.
```

Key rule: **Extraction rule must reference exact token.**

Never write vague extraction instructions like “find IDs in context.”

Ambiguity causes inconsistent judging.

---

## Criterion 1 — SCHEMA VALIDITY (Always CRITICAL)

```yaml
1) SCHEMA VALIDITY (CRITICAL)
- Must match {llm_schema} exactly.
- No extra fields anywhere.
- No missing required fields.
- All required string fields must be non-empty.
- Must not echo schema content.

If schema invalid → verdict=FAIL and score ≤ 30.
```

Schema failure is structural — score must be low.

---

# Additional Criteria (Task-Specific)

Add only criteria that generation prompt supports.

Common categories:

---

## FORMAT COMPLIANCE (CRITICAL if structural)

```yaml
N) FORMAT COMPLIANCE (CRITICAL)
For EACH item:
- Must follow required substructure.
- Must include required labeled sections.
- Must not include disallowed patterns.

If any violation → verdict=FAIL and score ≤ 50.
```

---

## ALIGNMENT TO INPUT (PER-ITEM)

```yaml
N) ALIGNMENT (PER-ITEM)
For EACH item:
- Must align with [input attribute].
- Must respect [constraint].
- Must not introduce out-of-scope content.

If multiple violations → score ≤ 60.
```

---

## UNIQUENESS (GROUP-LEVEL)

```yaml
N) UNIQUENESS (GROUP-LEVEL)
Each item must target a distinct dimension/category.

Explicit duplicate patterns:
- [Pattern A]
- [Pattern B]

Scoring:
- Near-duplicate pair → verdict=FAIL and score ≤ 60.
- Strong overlap across 2+ items → score ≤ 65.
```

Always define what constitutes duplication.

Generic uniqueness rules are unreliable.

---

## OUTPUT QUALITY (PER-ITEM)

```yaml
N) OUTPUT QUALITY (PER-ITEM)
For EACH item:
- Must meet [quality requirement].
- Must not be vague or trivial.
- Must not be shorter than [threshold].

If multiple items fail → score ≤ 70.
```

Be explicit about measurable failure conditions.

---

# Scoring Discipline (System Block)

LLMs inflate scores without constraint.

Include explicit bands:

```yaml
SCORING DISCIPLINE

90-100: Exceptional, fully compliant, strong coverage.
80-89: Strong, minor refinements possible.
70-79: Acceptable but noticeable weaknesses.
60-69: Major issue present.
40-59: Structural or alignment failure.
<40: Schema failure or severe break.

DO NOT give 90+ unless clearly exceptional.
```

Each criterion must define ceiling behavior.

Unbounded scoring → meaningless metrics.

---

# Diagnostic Reasons (Critical for Retry Loop)

Reasons are injected into next generation attempt.

They must be actionable.

Require:

```yaml
Reasons MUST be specific and diagnostic.
Include identifiers where possible.

Examples:
- "Missing IDs: [3,5]"
- "Duplicate ID: 4"
- "Field empty: rubric in item_id=2"
- "Near-duplicate dimensions: item_id=4 vs item_id=5"
```

Do not allow vague reasons:

* ❌ "Output is poor"
* ❌ "Low quality"

Vague reasons cause retry loops to stagnate.

---

# Generation ↔ Judge Alignment Rule

High retry rate almost always means misalignment.

Before finalizing judge.yaml:

* Every generation constraint must appear in judge criteria.
* Judge must not enforce unseen constraints.
* Score thresholds must reflect what generation can realistically produce.

Judge stricter than generation → perpetual retry.

Judge looser than generation → quality drift.

---

# Per-Item vs Group-Level Evaluation

For group mode, always explicitly separate:

| Level       | Checks                             |
| ----------- | ---------------------------------- |
| Group-level | Completeness, uniqueness, coverage |
| Per-item    | Schema, alignment, format, quality |

If not explicitly separated, the model often evaluates only one layer.

---

# Versioning

* `version` is human-only.
* Cache key uses SHA1 of file content.
* Editing prompt auto-invalidates cache.
* Rolling back restores cache hits.

Keep version history in guideline documentation, not in prompt file.

---

# Schema Synchronisation

Judge uses `{llm_schema}` to evaluate structural compliance.

If schema changes:

1. Run Pipeline 0
2. Run Pipeline 1
3. Confirm both generation + judge reference updated schema

Never manually edit `llm_schema.txt`.

---

# Tuning Guide

## High retry rate but outputs look fine

→ Generation and judge misaligned.

Fix criteria definitions before relaxing generation.

---

## Scores cluster too high

→ Strengthen scoring discipline.
→ Lower ceilings for known failure types.
→ Add “DO NOT give 90+ unless clearly exceptional.”

---

## Reasons not actionable

→ Add explicit diagnostic examples.
→ Require ID references.

---

## Completeness inconsistencies

→ Make identifier extraction rule exact.
→ Reference explicit context token.

---

## Judge regenerates instead of evaluating

→ Move "Do NOT regenerate" rule to very top.
→ Remove ambiguous language.
→ Ensure no instruction implies correction.

---

## Criterion never triggers

→ Failure condition too vague.
→ Replace subjective language with observable checks.

---

# Load-Bearing Elements (Do Not Change)

| Element         | Why                                   |
| --------------- | ------------------------------------- |
| `{llm_schema}`  | Required for schema compliance        |
| `{context}`     | Required for alignment + completeness |
| `{output_json}` | The object being evaluated            |
| `verdict`       | Retry gate reads this                 |
| `score`         | Manifest + reporting use this         |
| `reasons`       | Injected into retry                   |

Renaming any field requires coordinated changes to:

* Pipeline 4 runner
* Retry logic
* Pipeline 6 report generator

---

# Architectural Summary

`judge.yaml` defines:

* The formal acceptance criteria
* The retry trigger mechanism
* The scoring standard
* The feedback signal quality

It is not optional decoration.

It is a **deterministic quality gate**.

Its alignment with `generation.yaml` determines:

* Retry rate
* Artifact stability
* Output quality consistency
* Score meaning across runs

Treat it as a versioned specification document, not a prompt experiment.

---
