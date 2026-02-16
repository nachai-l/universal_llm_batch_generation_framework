# tests/test_processing.py
import json
from pathlib import Path

import pandas as pd
import pytest

from functions.core.processing import clean_string_columns, row_to_json, row_to_json_by_id


def test_clean_string_columns_basic():
    df = pd.DataFrame(
        {
            "a": ["  hello\nworld  ", "hello world", "x\t y", "\u00A0z\u00A0"],
            "b": [1, 2, 3, 4],
        }
    )

    out, stats = clean_string_columns(df)

    assert out.loc[0, "a"] == "hello world"
    assert out.loc[2, "a"] == "x y"
    assert out.loc[3, "a"] == "z"

    assert stats is not None
    # column 'a' should have fewer uniques after normalization in this example
    a_row = stats[stats["column"] == "a"].iloc[0]
    assert a_row["unique_before"] >= a_row["unique_after"]


def test_row_to_json_pretty_and_dict():
    df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})

    s = row_to_json(df, 0, pretty=True)
    obj = json.loads(s)
    assert obj == {"a": 1, "b": "x"}

    d = row_to_json(df, 0, pretty=False)
    assert d == {"a": 1, "b": "x"}


def test_row_to_json_null_handling_remove():
    df = pd.DataFrame({"a": [None], "b": ["x"]})
    d = row_to_json(df, 0, pretty=False, null_handling="remove")
    assert d == {"b": "x"}


def test_row_to_json_out_of_bounds():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        row_to_json(df, 99)


def test_row_to_json_by_id_exact_match():
    df = pd.DataFrame({"ID": ["abc", "def"], "x": [1, None]})

    d = row_to_json_by_id(df, "abc", pretty=False)
    assert d["ID"] == "abc"
    assert d["x"] == 1

    d2 = row_to_json_by_id(df, "def", pretty=False, null_handling="empty_string")
    assert d2["x"] == ""


def test_row_to_json_by_id_not_found():
    df = pd.DataFrame({"ID": ["abc"], "x": [1]})
    with pytest.raises(ValueError) as e:
        row_to_json_by_id(df, "zzz", pretty=False)
    assert "not found" in str(e.value).lower()


def test_row_to_json_by_id_multiple_hits():
    df = pd.DataFrame({"ID": ["abc", "abc"], "x": [1, 2]})
    with pytest.raises(ValueError):
        row_to_json_by_id(df, "abc", pretty=False)


def test_row_to_json_by_id_partial_match():
    df = pd.DataFrame({"ID": ["hello-123", "world-456"], "x": [1, 2]})
    d = row_to_json_by_id(df, "hello", pretty=False, allow_partial_match=True)
    assert d["ID"] == "hello-123"
