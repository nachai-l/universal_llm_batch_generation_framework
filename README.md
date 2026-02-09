# LLM Data Quality Exploration

LLM-based data quality exploration and validation framework that evaluates structured fields against raw text evidence, producing field-level reliability metrics, failure modes, and usability diagnostics.

---

## Overview

This project explores **real-world structured data quality** by validating each structured field against its original raw text source using **Large Language Models (LLMs)**.

Instead of assuming structured fields are correct, the framework treats raw text (e.g. job descriptions) as the source of truth and asks:

- Does this structured field actually match the evidence?
- Is the field missing, weakly inferred, or contradictory?
- Which fields are reliable enough for downstream use?
- Where should fallbacks or enrichment strategies be applied?

The result is a **diagnostic view of data trustworthiness**, not just completeness.

---

## Key Capabilities

- **Field-level validation**
  - Match / Unmatch / Unsure / NoData classification per field
- **Reliability scoring**
  - Standard and strict reliability metrics
- **Failure mode analysis**
  - Explicit separation of Unmatch vs Unsure vs NoData
- **Missingness dominance detection**
  - Identify fields failing primarily due to missing inputs
- **Record-level health metrics**
  - Problem density per record (Unmatch + Unsure)
- **Title usability diagnostics**
  - Structured title vs raw title fallback recommendation
- **LLM-extracted BODY skill analysis**
  - Skill frequency and per-record distribution
- **Traceable explanations**
  - Human-readable reasons for each Unmatch / Unsure decision

---

## Data Scope

- Dataset: Job postings (Lightcast sample)
- Sample size: Configurable (typically up to ~20,000 records depending on access and filters)
- Inputs:
  - Raw job text (`BODY`, `TITLE_RAW`, `COMPANY_RAW`, etc.)
  - Structured fields (title, salary, industry, occupation, location, education, etc.)
- Raw and structured sources are merged by record `ID`

---

## Project Structure

```

Lightcast_data_exploration/
├── configs/
│   ├── parameters.yaml        # Pipeline parameters
│   ├── prompts.yaml           # LLM prompt templates
│   └── credentials.yaml       # API credentials (ignored by git)
│
├── functions/
│   ├── batch/                 # Batch pipelines (pipeline_1, pipeline_2, ...)
│   ├── core/                  # Core computation logic
│   ├── llm/                   # LLM client & runner
│   ├── io/                    # Readers / writers
│   └── utils/                 # Shared utilities
│
├── raw_data/                  # Original raw CSV files (ignored by git)
├── process_data/              # Processed intermediate data (ignored by git)
│
├── artifacts/
│   ├── cache/                 # LLM response cache (+ failure dumps)
│   └── reports/               # Generated CSV & report artifacts
│
├── exploration.ipynb          # Interactive analysis notebook
├── requirements.txt
├── .gitignore
└── README.md

````

---

## Pipelines

### Pipeline 1 — LLM Field Validation

- Compares structured fields against raw job text
- Produces per-field validation results with explanations
- Enforces a **fixed, stable output schema**
- Supports caching, retries, concurrency, and cache bypass (`--force`)

**Outputs**
- `artifacts/job_postings_dq_eval.csv`
- `artifacts/job_postings_dq_eval.jsonl`

---

### Pipeline 2 — Aggregated Quality Report

- Aggregates Pipeline 1 outputs into analytical summaries
- Computes field-level and record-level quality diagnostics

**Produces**
- Overall quality metrics
- Field reliability tables
- Missingness dominance analysis
- Record health distributions
- BODY skill frequency and density reports

---

## Key Metrics Explained

| Metric | Meaning |
|------|--------|
| Match | Field is directly supported by raw text |
| Unmatch | Field contradicts raw text |
| Unsure | Raw text does not provide explicit evidence |
| NoData | Field missing or empty |
| Reliability Score | `1 − (unmatch_rate + unsure_rate)` |
| Strict Reliability | `1 − (unmatch_rate + unsure_rate + nodata_rate)` |
| Problem Rate | `unmatch_rate + unsure_rate` (excludes NoData) |

---

## Example Findings

- Salary information is present in ~17% of postings → **high NoData dominance**
- Title fields are highly reliable → **safe for direct use**
- Industry and education fields are often **Unsure** due to missing explicit mentions
- Most records have **30–50% problematic fields**, indicating partial usability rather than binary quality

---

## Design Philosophy

- **LLMs as validators, not generators**
- Raw text is the **source of truth**
- Structured data is treated as **hypotheses**
- Explicit uncertainty is better than false certainty
- Metrics must be interpretable by humans, not just models

---

## Configuration Notes

- `llm.max_rows_per_run`: limit rows per execution (`null` or `"all"` = full dataset)
- `llm.max_workers`: concurrency for LLM calls
- `--force`: bypass cache and re-run LLM evaluation
- Cached LLM outputs are stored in `artifacts/cache/`
- Invalid model outputs are dumped to `artifacts/cache/_failures/` for inspection

---

## Requirements

- Python 3.9+
- pandas
- PyYAML
- google-genai (Gemini SDK)

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## Notes

* Large raw datasets and credentials are intentionally excluded from version control.
* This framework is dataset-agnostic and can be adapted beyond job postings.
* Designed for **analysis, auditing, and decision support**, not model training.
