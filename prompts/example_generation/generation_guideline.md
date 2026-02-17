# Prompt Guideline: `generation.yaml`

**File:** `prompts/generation.yaml`
**Role in framework:** Drives Pipeline 4 LLM generation
**Mode compatibility:**

* Row-wise
* Group output
* Row output with group context

---

## What This Prompt Does

`generation.yaml` is the **primary generation prompt** used by Pipeline 4.

For each WorkItem, the framework:

1. Renders this prompt with runtime placeholders injected
2. Sends it to the LLM
3. Validates the response strictly using Pydantic v2
4. Optionally passes it through a judge prompt
5. Persists only validated (and judge-approved, if enabled) outputs

The model must return a **single structured JSON object** that passes strict runtime validation.

This prompt defines:

* The **output contract**
* The **task instructions**
* The injection points for schema and context
* Task-specific structural and semantic constraints

---

## Prompt Structure

Each `generation.yaml` must contain:

| Block     | Purpose                                                  |
| --------- | -------------------------------------------------------- |
| `name`    | Human-readable identifier                                |
| `version` | Manual version tracking (not used for cache keying)      |
| `purpose` | Short description of task                                |
| `system`  | Hard output rules + structural constraints               |
| `user`    | Task instructions + schema injection + context injection |

### Separation of Responsibilities

* `system` block → **how to output**
* `user` block → **what to generate**

Do not mix these responsibilities. Keeping them separate makes prompt tuning easier and reduces unintended side effects.

---

## Runtime Placeholders

The framework injects placeholders by exact string match.

### Required placeholders

| Placeholder    | Injected from           | Contains                      |
| -------------- | ----------------------- | ----------------------------- |
| `{llm_schema}` | `schema/llm_schema.txt` | Full JSON schema contract     |
| `{context}`    | Pipeline 3 WorkItem     | Input rows for this work unit |

These placeholders **must not be renamed or removed**.

### Context shape depends on grouping mode

* **Row-wise mode:** `{context}` contains a single input row.
* **Group output mode:** `{context}` contains multiple rows in the same group.
* **Row output with group context mode:** `{context}` contains one target row plus sibling rows within group context.

The prompt should not assume a specific shape beyond what the task requires.

### Forward compatibility note

These are the required placeholders today. The framework may introduce additional placeholders in the future (e.g., run metadata, retry feedback, group identifiers). Prompts should not assume `{llm_schema}` and `{context}` are the only possible injections.

---

## Output Contract (System Block)

The `system` block must clearly define the hard output contract.

Recommended minimum:

```yaml
system: |
  HARD OUTPUT CONTRACT
  - Output MUST be valid JSON matching {llm_schema} exactly.
  - Return ONE JSON object only.
  - No markdown.
  - No commentary.
  - No explanations.
  - No code fences.
  - Do NOT echo the schema.
  - Do NOT include fields not defined in the schema.
  - All required fields must be present.
```

Optional (only if schema and judge expect it):

```
  - All required string fields must be non-empty.
```

Only include non-empty requirements if your schema or judge enforces it. Otherwise, do not over-constrain.

---

## Why This Matters

Pipeline 4 performs strict validation using:

* Pydantic v2
* `extra="forbid"`

Any deviation (markdown, commentary, array instead of object, extra fields) causes validation failure and retry.

Common violations to guard against:

| Violation      | Guard                                             |
| -------------- | ------------------------------------------------- |
| Code fences    | “No code fences”                                  |
| Array returned | “Return ONE JSON object only”                     |
| Schema echoed  | “Do NOT echo the schema”                          |
| Extra fields   | “Do NOT include fields not defined in the schema” |

---

## Task Instructions (User Block)

The `user` block should follow this structure:

1. Context overview
2. Task definition
3. Field-level guidance
4. Quality rules (if any)
5. Schema injection (`{llm_schema}`)
6. Context injection (`{context}`)
7. Final reminder

Example structure:

```yaml
user: |
  TASK OVERVIEW
  ...

  OUTPUT REQUIREMENTS
  ...

  FIELD-LEVEL GUIDANCE
  - field_a: ...
  - field_b: ...

  OUTPUT SCHEMA
  {llm_schema}

  INPUT CONTEXT
  {context}

  FINAL REMINDER
  Return only the JSON object.
```

Place `{llm_schema}` and `{context}` near the end to ensure the model sees them immediately before generation.

---

## Internal Validation (Recommended)

For strict tasks, add internal validation instructions:

```yaml
system: |
  INTERNAL VALIDATION BEFORE RETURNING
  For EACH output item:
  - Verify [constraint 1]
  - Verify [constraint 2]
  - Verify [constraint 3]
  If any check fails, revise internally before returning JSON.
```

This reduces judge retry rate significantly.

---

## Internal Planning (Recommended for Group Output)

For `group_output` mode:

```yaml
system: |
  PLANNING (INTERNAL — DO NOT OUTPUT)
  Before generating:
  1. Identify all required items.
  2. Assign a unique dimension/category to each.
  3. Verify coverage diversity.
  4. Then generate.
```

This prevents repetitive outputs.

---

## Quality Rules

Only add rules that the judge enforces.

Examples:

**Uniqueness rule**

```
- Each item must target a different dimension.
- No rewording of the same idea.
```

**Format rule**

```
- Must begin with ...
- Must end with ...
- Must be at least 50 words.
```

**Scope rule**

```
- Limit to junior-level complexity.
- Do NOT include advanced system design.
```

Unenforced rules increase prompt length without improving reliability.

---

## Field-Level Guidance

For structured fields:

```yaml
- rubric: Must contain three clearly separated sections:
    Section A:
    Section B:
    Section C:
```

If structure is critical, include a minimal format example.

Negative examples (“BAD: …”) often improve compliance more than positive examples alone.

---

## Versioning

The `version` field is for human tracking only.

Cache keying uses SHA1 of prompt file content, not version number.

Implications:

* Editing prompt text → new hash → automatic cache miss
* Rolling back file content restores previous cache hits

Recommended version history table:

| Version | Change                       |
| ------- | ---------------------------- |
| 1       | Initial                      |
| 2       | Strengthened output contract |
| 3       | Added internal planning      |

Increment version when making intentional changes.

---

## Schema Synchronization

`llm_schema.txt` is derived from `llm_schema.py` via Pipeline 0 + 1.

Never manually edit `llm_schema.txt`.

If validation failures suggest drift:

1. Re-run Pipeline 0
2. Re-run Pipeline 1
3. Confirm schema regeneration

Prompt contract must always match runtime schema.

---

## Tuning Guide

### If structure is wrong

* Move HARD OUTPUT CONTRACT to very top of `system`
* Remove competing language

### If outputs are repetitive

* Add uniqueness rules
* Add internal planning step

### If fields are vague

* Add field-level instructions
* Provide format examples
* Add negative examples

### If judge retry rate is high

Most common cause:
Generation prompt and judge prompt are misaligned.

Ensure judge criteria are reflected explicitly in generation instructions.

---

## Load-Bearing Elements (Do Not Change)

| Element                      | Why                                        |
| ---------------------------- | ------------------------------------------ |
| `{llm_schema}`               | Required for validation                    |
| `{context}`                  | Required for input injection               |
| Single top-level JSON object | Assumed by current Pipeline 5 export logic |
| Schema field names           | Must match `llm_schema.py` exactly         |

Structural changes (e.g., switching to array output) require coordinated updates to:

* `llm_schema.py`
* Pipeline 0 + 1 regeneration
* Pipeline 5 export logic

---

## Summary

`generation.yaml` is not just a prompt. It is a **contract layer** between:

* Input data (Pipeline 3)
* Schema validation (Pipeline 4)
* Export logic (Pipeline 5)
* Judge gating (optional)

Treat it as a versioned, controlled interface — not as an ad-hoc prompt.

Any structural change must be coordinated across schema, export logic, and judge criteria.

---
