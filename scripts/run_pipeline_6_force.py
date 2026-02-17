# scripts/run_pipeline_6_force.py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.batch.pipeline_6_write_report import main
from functions.utils.logging import get_logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("=" * 80)
    logger.info("RUNNING PIPELINE 6 — WRITE REPORT (MD + optional HTML)")
    logger.info("=" * 80)

    rc = main()

    logger.info("Pipeline 6 finished with return code: %d", int(rc))
    logger.info("=" * 80)
    raise SystemExit(rc)
