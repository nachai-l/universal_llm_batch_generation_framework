# functions/batch/pipeline_3_build_requests.py
"""
Pipeline 3 — Build Requests (WorkItems)

Intent
------
Turn the ingested input table (Pipeline 2 artifact) into deterministic WorkItems
ready for LLM execution (Pipeline 4).

Design
------
- Pipeline stays THIN: orchestration + I/O only.
- Main logic uses core/context_builder:
    - build_work_items() (legacy / non-dedup modes)
    - build_work_items_and_group_contexts(..., dedupe_group_context=True) (dedup mode)
- Output is a deterministic JSON artifact used by later pipelines.

Inputs
------
- configs/parameters.yaml
- artifacts/cache/pipeline2_input.json

Outputs
-------
- artifacts/cache/pipeline3_work_items.json
- artifacts/cache/pipeline3_group_contexts.json (ONLY when:
    grouping.enabled=true AND grouping.mode=row_output_with_group_context)

Notes
-----
- Legacy behavior:
    - WorkItem.context is the final string injected into prompts as {context}.
- Dedup behavior (2026-02):
    - We de-duplicate group context blobs for row_output_with_group_context to avoid
      repeating the same context N times (huge artifact size at scale).
    - Pipeline writes a separate group context artifact:
        pipeline3_group_contexts.json  (list of GroupContext)
    - Each WorkItem stores:
        group_key + group_context_id (in meta)
      and WorkItem.context is an empty string.
- WorkItem.work_id is deterministic and becomes the natural cache_id seed for Pipeline 4.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from functions.core.context_builder import (
    GroupContext,
    WorkItem,
    build_work_items_and_group_contexts,
)
from functions.io.readers import read_json
from functions.io.writers import write_json
from functions.utils.config import load_parameters
from functions.utils.logging import get_logger


PIPELINE2_INGEST_PATH_DEFAULT = "artifacts/cache/pipeline2_input.json"
PIPELINE3_WORK_ITEMS_PATH_DEFAULT = "artifacts/cache/pipeline3_work_items.json"
PIPELINE3_GROUP_CONTEXTS_PATH_DEFAULT = "artifacts/cache/pipeline3_group_contexts.json"


def _work_item_to_dict(item: WorkItem) -> Dict[str, Any]:
    """
    Convert WorkItem dataclass to JSON-serializable dict deterministically.
    """
    # asdict is deterministic for dataclasses; nested dict ordering stabilized by writers.write_json(sort_keys=True)
    d = asdict(item)

    # Ensure stable types
    d["group_key"] = "" if d.get("group_key") is None else str(d.get("group_key"))
    d["row_index"] = None if d.get("row_index") is None else int(d.get("row_index"))

    # Guarantee meta is a dict (forward-compat)
    if d.get("meta") is None:
        d["meta"] = {}
    return d


def _group_context_to_dict(gc: GroupContext) -> Dict[str, Any]:
    """
    Convert GroupContext dataclass to JSON-serializable dict deterministically.
    """
    d = asdict(gc)
    d["group_key"] = "" if d.get("group_key") is None else str(d.get("group_key"))
    d["group_context_id"] = str(d.get("group_context_id") or "")
    if d.get("meta") is None:
        d["meta"] = {}
    return d


def _should_dedupe_group_context(params: Any) -> bool:
    """
    Dedupe applies only for:
      grouping.enabled=true AND grouping.mode=row_output_with_group_context
    """
    enabled = bool(getattr(getattr(params, "grouping", None), "enabled", False))
    mode = getattr(getattr(params, "grouping", None), "mode", None)
    return bool(enabled and mode == "row_output_with_group_context")


def _build_output_payload(
    *,
    ingest: Dict[str, Any],
    items: List[WorkItem],
    group_contexts: Optional[List[GroupContext]],
    params: Any,
) -> Dict[str, Any]:
    meta_in = ingest.get("meta", {}) if isinstance(ingest, dict) else {}
    dedup = _should_dedupe_group_context(params)

    out: Dict[str, Any] = {
        "meta": {
            "source": {
                "pipeline2_ingest_path": meta_in.get("input_path") or meta_in.get("source_path") or None,
                "n_rows": meta_in.get("n_rows"),
                "n_cols": meta_in.get("n_cols"),
                "columns": meta_in.get("columns"),
            },
            "grouping": {
                "enabled": bool(getattr(params.grouping, "enabled", False)),
                "column": getattr(params.grouping, "column", None),
                "mode": getattr(params.grouping, "mode", None),
                "max_rows_per_group": getattr(params.grouping, "max_rows_per_group", None),
                "dedupe_group_context": dedup,
            },
            "context": {
                "columns_mode": getattr(getattr(params.context, "columns", None), "mode", None)
                if getattr(params, "context", None) is not None
                else None,
                "kv_order": getattr(getattr(params, "context", None), "kv_order", None),
                "max_context_chars": getattr(getattr(params, "context", None), "max_context_chars", None),
                "truncate_field_chars": getattr(getattr(params, "context", None), "truncate_field_chars", None),
                "row_template": getattr(getattr(params, "context", None), "row_template", None),
                "group_header_template": getattr(getattr(params, "context", None), "group_header_template", None),
                "group_footer_template": getattr(getattr(params, "context", None), "group_footer_template", None),
            },
        },
        "n_items": len(items),
        "items": [_work_item_to_dict(x) for x in items],
    }

    # Keep the main work_items artifact self-describing by including counts + pointer path
    if dedup:
        out["n_group_contexts"] = len(group_contexts or [])
        out["group_contexts_artifact"] = PIPELINE3_GROUP_CONTEXTS_PATH_DEFAULT

    return out


def main(
    *,
    parameters_path: str | Path = "configs/parameters.yaml",
    ingest_path: str | Path = PIPELINE2_INGEST_PATH_DEFAULT,
    out_path: str | Path = PIPELINE3_WORK_ITEMS_PATH_DEFAULT,
    out_group_contexts_path: str | Path = PIPELINE3_GROUP_CONTEXTS_PATH_DEFAULT,
) -> int:
    logger = get_logger(__name__)

    params = load_parameters(parameters_path)
    ingest = read_json(ingest_path)

    records = ingest.get("records") or ingest.get("rows") or []
    if not isinstance(records, list):
        raise ValueError(f"pipeline2 ingest artifact malformed: records is not a list: {type(records)}")

    if len(records) == 0:
        payload = _build_output_payload(ingest=ingest, items=[], group_contexts=[], params=params)
        write_json(out_path, payload)
        logger.info("Pipeline 3 completed: 0 items (empty input)")
        return 0

    df = pd.DataFrame.from_records(records)

    dedup = _should_dedupe_group_context(params)

    if dedup:
        items, group_contexts = build_work_items_and_group_contexts(
            df=df,
            params=params,
            dedupe_group_context=True,
        )
        # Write group contexts artifact (dedup mode only)
        write_json(out_group_contexts_path, [_group_context_to_dict(x) for x in group_contexts])

        payload = _build_output_payload(
            ingest=ingest,
            items=items,
            group_contexts=group_contexts,
            params=params,
        )
        write_json(out_path, payload)

        logger.info(
            "Pipeline 3 completed (DEDUP): wrote %d WorkItems to %s and %d GroupContexts to %s",
            len(items),
            str(out_path),
            len(group_contexts),
            str(out_group_contexts_path),
        )
        return 0

    # Non-dedup path (legacy behavior preserved)
    items, _ = build_work_items_and_group_contexts(
        df=df,
        params=params,
        dedupe_group_context=False,
    )

    payload = _build_output_payload(
        ingest=ingest,
        items=items,
        group_contexts=None,
        params=params,
    )
    write_json(out_path, payload)

    logger.info(
        "Pipeline 3 completed: wrote %d WorkItems to %s",
        len(items),
        str(out_path),
    )
    return 0


__all__ = ["main"]
