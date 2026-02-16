# scripts/run_pipeline_2_force.py
"""
Manual runner — Pipeline 2 (REAL execution)

Usage:
    python scripts/run_pipeline_2_force.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.batch.pipeline_2_ingest_input import main as pipeline_2_main
from functions.utils.config import load_parameters
from functions.utils.logging import get_logger


def main() -> int:
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("RUNNING PIPELINE 2 — REAL EXECUTION")
    logger.info("Repo root: %s", REPO_ROOT)
    logger.info("Working directory: %s", Path.cwd())
    logger.info("=" * 80)

    required_files = [
        REPO_ROOT / "configs/parameters.yaml",
        REPO_ROOT / "configs/credentials.yaml",
    ]
    for p in required_files:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    params = load_parameters()
    inp = REPO_ROOT / str(params.input.path)
    if not inp.exists():
        raise FileNotFoundError(f"Input file missing: {inp}")

    logger.info("Invoking pipeline_2_ingest_input.main()")
    rc = pipeline_2_main()

    logger.info("Pipeline 2 finished with return code: %s", rc)
    logger.info("=" * 80)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
