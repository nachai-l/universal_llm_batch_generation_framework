# functions/batch/pipeline_1_schema_txt_ensure.py
"""
Pipeline 1 — Ensure schema/llm_schema.txt exists.

Behavior
- If txt exists and llm_schema.force_regenerate=false: no-op.
- If txt missing: generate from schema/llm_schema.py deterministically.
- If force_regenerate=true: archive old txt and regenerate.

Preferred output:
- JSON Schema (text) derived from Pydantic models via model_json_schema()
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from functions.core.schema_text import build_llm_schema_txt_from_py_file
from functions.utils.config import ensure_dirs, load_parameters
from functions.utils.logging import get_logger


def _archive_existing(txt_path: Path, archive_dir: Path) -> Optional[Path]:
    if not txt_path.exists():
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_path = archive_dir / f"{txt_path.stem}_{ts}{txt_path.suffix}"
    shutil.copy2(txt_path, archived_path)
    return archived_path


def main() -> int:
    logger = get_logger(__name__)

    params = load_parameters()
    ensure_dirs(params)

    py_path = Path(params.llm_schema.py_path)
    txt_path = Path(params.llm_schema.txt_path)
    archive_dir = Path(params.llm_schema.archive_dir)

    force_regen = bool(getattr(params.llm_schema, "force_regenerate", False))

    if txt_path.exists() and not force_regen:
        logger.info("Pipeline 1: schema txt exists; no-op. path=%s", txt_path)
        return 0

    if not py_path.exists():
        raise RuntimeError(
            f"Pipeline 1 requires schema py to exist, but it is missing: {py_path}. "
            "Run Pipeline 0 first."
        )

    if txt_path.exists() and force_regen:
        archived = _archive_existing(txt_path, archive_dir)
        logger.info("Archived existing schema txt to: %s", archived)

    txt = build_llm_schema_txt_from_py_file(py_path)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(txt, encoding="utf-8")

    logger.info("Wrote schema txt to: %s (bytes=%s)", txt_path, txt_path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
