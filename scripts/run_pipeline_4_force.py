# scripts/run_pipeline_4_force.py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.batch.pipeline_4_llm_generate import main
from functions.utils.logging import get_logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("=" * 80)
    logger.info("RUNNING PIPELINE 4 — REAL EXECUTION (FORCE)")
    logger.info("=" * 80)

    # Force re-generation by passing a temp parameters override file is overkill here;
    # this script assumes you set cache.force: true in configs/parameters.yaml when needed.
    rc = main()

    logger.info("Pipeline 4 finished with return code: %d", int(rc))
    logger.info("=" * 80)
    raise SystemExit(rc)
