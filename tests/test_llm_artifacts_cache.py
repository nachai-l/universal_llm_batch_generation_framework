# tests/test_llm_artifacts_cache.py
from __future__ import annotations

import pytest

from functions.core.llm_artifacts_cache import load_group_context_map


def test_load_group_context_map_accepts_top_level_list():
    obj = [
        {"group_key": "A", "group_context_id": "gid_A", "context": "ctx_A"},
        {"group_key": "B", "group_context_id": "gid_B", "context": "ctx_B"},
    ]
    m = load_group_context_map(obj)
    assert m == {"gid_A": "ctx_A", "gid_B": "ctx_B"}


def test_load_group_context_map_accepts_groups_key_dict():
    obj = {
        "n_groups": 2,
        "groups": [
            {"group_key": "A", "group_context_id": "gid_A", "context": "ctx_A"},
            {"group_key": "B", "group_context_id": "gid_B", "context": "ctx_B"},
        ],
    }
    m = load_group_context_map(obj)
    assert m == {"gid_A": "ctx_A", "gid_B": "ctx_B"}


def test_load_group_context_map_accepts_direct_mapping_dict():
    obj = {
        "gid_A": {"context": "ctx_A"},
        "gid_B": {"context": "ctx_B"},
    }
    m = load_group_context_map(obj)
    assert m == {"gid_A": "ctx_A", "gid_B": "ctx_B"}


def test_load_group_context_map_raises_on_invalid_type():
    with pytest.raises(ValueError):
        load_group_context_map("not-a-dict-or-list")


def test_load_group_context_map_raises_on_empty_groups_list():
    with pytest.raises(ValueError):
        load_group_context_map([])
