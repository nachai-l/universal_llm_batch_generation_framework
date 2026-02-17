# tests/test_context_builder.py

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from functions.core.context_builder import (
    GroupContext,
    WorkItem,
    build_work_items,
    build_work_items_and_group_contexts,
)


def _params(
    *,
    grouping_enabled: bool = False,
    grouping_column: str | None = None,
    grouping_mode: str = "group_output",
    max_rows_per_group: int = 50,
    # forward-compat context block
    columns_mode: str = "all",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    row_template: str = "{__ROW_KV_BLOCK__}",
    auto_kv_block: bool = True,
    kv_order: str = "input_order",
    max_context_chars: int = 0,
    truncate_field_chars: int = 0,
):
    include = include or []
    exclude = exclude or []

    # mimic your typed ParametersConfig shape (attrs, not dict)
    params = SimpleNamespace(
        grouping=SimpleNamespace(
            enabled=grouping_enabled,
            column=grouping_column,
            mode=grouping_mode,
            max_rows_per_group=max_rows_per_group,
        ),
        # your ParametersConfig doesn't have context yet; we add it here to test forward-compat
        context=SimpleNamespace(
            columns=SimpleNamespace(
                mode=columns_mode,
                include=include,
                exclude=exclude,
            ),
            row_template=row_template,
            auto_kv_block=auto_kv_block,
            kv_order=kv_order,
            max_context_chars=max_context_chars,
            truncate_field_chars=truncate_field_chars,
        ),
    )
    return params


def _df_basic() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"group": "A", "q": "hello", "note": "  spaced   text  "},
            {"group": "A", "q": "world", "note": None},
            {"group": "B", "q": "foo", "note": "bar"},
        ]
    )


def test_empty_df_returns_empty_list():
    df = pd.DataFrame([])
    params = _params()
    items = build_work_items(df, params)
    assert items == []


def test_rowwise_builds_one_item_per_row():
    df = _df_basic()
    params = _params(grouping_enabled=False)
    items = build_work_items(df, params)

    assert len(items) == len(df)
    assert all(isinstance(x, WorkItem) for x in items)

    # row_index and group_key behavior
    assert [x.row_index for x in items] == [0, 1, 2]
    assert all(x.group_key is None for x in items)

    # deterministic work_id uniqueness
    assert len({x.work_id for x in items}) == len(items)


def test_rowwise_respects_column_exclude():
    df = _df_basic()
    params = _params(grouping_enabled=False, columns_mode="exclude", exclude=["note"])
    items = build_work_items(df, params)

    assert len(items) == len(df)
    # "note:" should not appear
    assert all("note:" not in x.context for x in items)
    # still includes other columns
    assert all("q:" in x.context for x in items)


def test_rowwise_kv_order_alpha_sorts_columns():
    df = _df_basic()[["note", "q", "group"]]  # intentionally shuffled
    params = _params(grouping_enabled=False, kv_order="alpha")
    items = build_work_items(df, params)

    # For first row, alpha order should be: group, note, q
    first = items[0].context.splitlines()
    assert first[0].startswith("group:")
    assert first[1].startswith("note:")
    assert first[2].startswith("q:")


def test_field_truncation_applies_and_marks_meta():
    df = _df_basic()
    params = _params(grouping_enabled=False, truncate_field_chars=5)
    items = build_work_items(df, params)

    meta0 = items[0].meta["kv"]["field_truncated"]
    assert meta0["note"] is True
    assert meta0["q"] is False

    # ellipsis present on truncated field value
    assert "note:" in items[0].context
    assert "…" in items[0].context


def test_context_truncation_applies_and_marks_meta():
    df = _df_basic()
    params = _params(grouping_enabled=False, max_context_chars=10)
    items = build_work_items(df, params)

    assert all(x.meta["context_truncation"]["applied"] is True for x in items)
    assert all(len(x.context) <= 11 for x in items)  # 10 + ellipsis possible


def test_group_output_builds_one_item_per_group_in_input_order():
    df = _df_basic()
    params = _params(
        grouping_enabled=True,
        grouping_column="group",
        grouping_mode="group_output",
    )
    items = build_work_items(df, params)

    # groups in first-seen order: A then B
    assert len(items) == 2
    assert items[0].group_key == "A"
    assert items[1].group_key == "B"
    assert items[0].row_index is None
    assert items[1].row_index is None

    # ✅ NEW: group_output now carries a deterministic group_context_id
    assert isinstance(items[0].meta.get("group_context_id"), str) and items[0].meta["group_context_id"]
    assert isinstance(items[1].meta.get("group_context_id"), str) and items[1].meta["group_context_id"]
    assert items[0].meta["group_context_id"] != items[1].meta["group_context_id"]

    # group context concatenates rows with blank lines
    assert "\n\n" in items[0].context
    assert "q: hello" in items[0].context
    assert "q: world" in items[0].context
    assert "q: foo" in items[1].context


def test_group_output_respects_max_rows_per_group_cap():
    df = _df_basic()
    params = _params(
        grouping_enabled=True,
        grouping_column="group",
        grouping_mode="group_output",
        max_rows_per_group=1,
    )
    items = build_work_items(df, params)

    a_ctx = items[0].context
    # only first A row should be included
    assert "q: hello" in a_ctx
    assert "q: world" not in a_ctx
    assert items[0].meta["row_cap_applied"] is True
    assert len(items[0].meta["row_indices_used"]) == 1

    # ✅ NEW: still has group_context_id even when capped
    assert isinstance(items[0].meta.get("group_context_id"), str) and items[0].meta["group_context_id"]


def test_row_output_with_group_context_builds_one_item_per_row_with_group_context_legacy():
    """
    Legacy behavior: WorkItem.context contains full group context repeated.
    """
    df = _df_basic()
    params = _params(
        grouping_enabled=True,
        grouping_column="group",
        grouping_mode="row_output_with_group_context",
    )
    items = build_work_items(df, params)

    assert len(items) == len(df)
    assert [x.row_index for x in items] == [0, 1, 2]
    assert [x.group_key for x in items] == ["A", "A", "B"]

    # rows 0 and 1 share the same group context (A), which includes both A rows
    assert items[0].context == items[1].context
    assert "q: hello" in items[0].context
    assert "q: world" in items[0].context

    # row 2 has group B context only
    assert "q: foo" in items[2].context
    assert "q: hello" not in items[2].context


def test_row_output_with_group_context_deduped_emits_group_contexts_and_references_only():
    """
    New behavior: de-dup group contexts.
    WorkItems reference group_context_id; WorkItem.context is empty.
    """
    df = _df_basic()
    params = _params(
        grouping_enabled=True,
        grouping_column="group",
        grouping_mode="row_output_with_group_context",
    )

    items, group_contexts = build_work_items_and_group_contexts(df, params, dedupe_group_context=True)

    assert len(items) == len(df)
    assert all(isinstance(x, WorkItem) for x in items)
    assert all(isinstance(g, GroupContext) for g in group_contexts)

    # one group context per group (A,B)
    assert len(group_contexts) == 2
    gk_set = {g.group_key for g in group_contexts}
    assert gk_set == {"A", "B"}

    # WorkItems have no repeated context
    assert all(x.context == "" for x in items)
    assert all(x.meta.get("group_context_id") for x in items)
    assert all(x.meta.get("deduped_group_context") is True for x in items)

    # rows 0 and 1 (group A) reference the same group_context_id
    gid0 = items[0].meta["group_context_id"]
    gid1 = items[1].meta["group_context_id"]
    gid2 = items[2].meta["group_context_id"]
    assert gid0 == gid1
    assert gid2 != gid0


def test_group_context_id_is_deterministic_for_same_input():
    df = _df_basic()
    params = _params(
        grouping_enabled=True,
        grouping_column="group",
        grouping_mode="row_output_with_group_context",
    )

    _items1, gcs1 = build_work_items_and_group_contexts(df, params, dedupe_group_context=True)
    _items2, gcs2 = build_work_items_and_group_contexts(df, params, dedupe_group_context=True)

    # Compare by group_key -> id
    m1 = {g.group_key: g.group_context_id for g in gcs1}
    m2 = {g.group_key: g.group_context_id for g in gcs2}
    assert m1 == m2


# ✅ NEW CASE 1: group_output group_context_id determinism (same df + same params)
def test_group_output_group_context_id_is_deterministic_for_same_input():
    df = _df_basic()
    params = _params(
        grouping_enabled=True,
        grouping_column="group",
        grouping_mode="group_output",
        max_rows_per_group=50,
    )

    items1 = build_work_items(df, params)
    items2 = build_work_items(df, params)

    m1 = {it.group_key: it.meta.get("group_context_id") for it in items1}
    m2 = {it.group_key: it.meta.get("group_context_id") for it in items2}
    assert m1 == m2
    assert set(m1.keys()) == {"A", "B"}
    assert all(isinstance(v, str) and v for v in m1.values())


# ✅ NEW CASE 2: group_output group_context_id changes when group context changes (cap affects context)
def test_group_output_group_context_id_changes_when_max_rows_per_group_changes():
    df = _df_basic()

    params_full = _params(
        grouping_enabled=True,
        grouping_column="group",
        grouping_mode="group_output",
        max_rows_per_group=50,  # includes both A rows
    )
    params_capped = _params(
        grouping_enabled=True,
        grouping_column="group",
        grouping_mode="group_output",
        max_rows_per_group=1,  # only first A row
    )

    items_full = build_work_items(df, params_full)
    items_capped = build_work_items(df, params_capped)

    # Compare A only (B unchanged by cap in this dataset)
    a_full = next(x for x in items_full if x.group_key == "A")
    a_cap = next(x for x in items_capped if x.group_key == "A")

    assert "q: world" in a_full.context
    assert "q: world" not in a_cap.context

    # If context changes, group_context_id should change
    assert a_full.meta["group_context_id"] != a_cap.meta["group_context_id"]


def test_grouping_enabled_missing_column_raises():
    df = _df_basic()
    params = _params(
        grouping_enabled=True,
        grouping_column="missing",
        grouping_mode="group_output",
    )
    with pytest.raises(ValueError):
        build_work_items(df, params)
