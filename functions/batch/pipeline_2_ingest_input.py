# functions/batch/pipeline_2_ingest_input.py
"""
Pipeline 2 — Ingest input table and write a readable JSON artifact.

Intent
- Load the configured input table (csv/tsv/psv/xlsx).
- Normalize data deterministically (string-only, NaN handled upstream).
- Emit a readable JSON artifact for:
  - debugging
  - inspection
  - downstream deterministic pipelines

Design decisions
- Data types: read as str for determinism
- Missing cells: normalized upstream (no pandas NaN leakage)
- Artifact format: JSON (readable, pretty-printed)

Output
- {params.cache.dir}/pipeline2_input.json
"""

from __future__ import annotations

from pathlib import Path

from functions.core.ingestions import ingest_input_table_pipeline2
from functions.io.writers import write_json
from functions.utils.config import ensure_dirs, load_parameters
from functions.utils.logging import get_logger


def main() -> int:
    logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Load config + ensure filesystem layout
    # ------------------------------------------------------------------
    params = load_parameters()
    ensure_dirs(params)

    # ------------------------------------------------------------------
    # Ingest input table (pure, deterministic core logic)
    # Returns a JSON-serializable payload (dict / list / primitives only)
    # ------------------------------------------------------------------
    payload = ingest_input_table_pipeline2(params)

    # ------------------------------------------------------------------
    # Write readable JSON artifact
    # ------------------------------------------------------------------
    out_path = Path(params.cache.dir) / "pipeline2_input.json"
    write_json(out_path, payload, indent=2, sort_keys=True)

    logger.info("Pipeline 2 completed: wrote ingest artifact: %s", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
