import json
import pandas as pd
import pytest

from functions.core.export_outputs import (
    select_question_for_row,
    flatten_for_psv,
    sort_key_for_export,
)


# ============================================================
# select_question_for_row
# ============================================================

def test_select_question_for_row_match():
    parsed_group = {
        "questions": [
            {"question_id": "1", "question_name": "Q1"},
            {"question_id": "2", "question_name": "Q2"},
        ]
    }

    input_row = {"Question ID": "2"}

    result = select_question_for_row(parsed_group, input_row)

    assert result["question_id"] == "2"
    assert result["question_name"] == "Q2"


def test_select_question_for_row_no_match():
    parsed_group = {
        "questions": [{"question_id": "1"}]
    }

    input_row = {"Question ID": "99"}

    result = select_question_for_row(parsed_group, input_row)

    assert result == {}


def test_select_question_for_row_missing_questions():
    parsed_group = {}

    input_row = {"Question ID": "1"}

    result = select_question_for_row(parsed_group, input_row)

    assert result == {}


# ============================================================
# flatten_for_psv
# ============================================================

def test_flatten_for_psv_basic():
    record = {
        "input": {"Role": "Data Scientist", "Question ID": "1"},
        "parsed": {"question_name": "Test Question"},
        "judge": {"verdict": "PASS", "score": 95},
        "meta": {"group_key": "Data Scientist"},
    }

    flat = flatten_for_psv(record, input_columns=["Role", "Question ID"])

    assert flat["Role"] == "Data Scientist"
    assert flat["Question ID"] == "1"
    assert flat["question_name"] == "Test Question"
    assert flat["judge_verdict"] == "PASS"
    assert flat["judge_score"] == 95


def test_flatten_for_psv_no_group_debug_columns():
    """
    Ensure removed heavy debug columns are NOT present.
    """
    record = {
        "input": {"Role": "Data Scientist"},
        "parsed": {"question_name": "Q"},
        "meta": {},
    }

    flat = flatten_for_psv(record, input_columns=["Role"])

    forbidden = {
        "group_rows_json",
        "questions_json",
        "group_context",
        "group_context_meta_json",
    }

    for col in forbidden:
        assert col not in flat


def test_flatten_for_psv_sanitizes_newlines():
    record = {
        "input": {"Role": "Data\nScientist"},
        "parsed": {"question_name": "Line1\nLine2"},
        "meta": {},
    }

    flat = flatten_for_psv(record, input_columns=["Role"])

    assert "\\n" in flat["Role"]
    assert "\\n" in flat["question_name"]


# ============================================================
# sort_key_for_export
# ============================================================

def test_sort_key_for_export_with_row_index():
    rec = {
        "meta": {
            "row_index": 5,
            "group_key": "A",
            "work_id": "xyz"
        }
    }

    key = sort_key_for_export(rec)

    assert key[0] == 5
    assert key[1] == "A"
    assert key[2] == "xyz"


def test_sort_key_for_export_no_row_index():
    rec = {
        "meta": {
            "group_key": "B",
            "work_id": "abc"
        }
    }

    key = sort_key_for_export(rec)

    # large sentinel for missing row_index
    assert key[0] > 10**6
    assert key[1] == "B"
    assert key[2] == "abc"


# ============================================================
# PSV deterministic column ordering integration-style
# ============================================================

def test_psv_dataframe_column_order_stable():
    records = [
        {
            "input": {"Role": "A", "Question ID": "1"},
            "parsed": {"question_name": "Q1"},
            "judge": {"verdict": "PASS", "score": 90},
            "meta": {"group_key": "A"},
        }
    ]

    flat = [flatten_for_psv(r, input_columns=["Role", "Question ID"]) for r in records]
    df = pd.DataFrame(flat)

    expected_prefix = ["Role", "Question ID"]

    for col in expected_prefix:
        assert col in df.columns

    # parsed column present
    assert "question_name" in df.columns

    # judge columns present
    assert "judge_verdict" in df.columns
    assert "judge_score" in df.columns


# ============================================================
# Deterministic stringify behavior
# ============================================================

def test_flatten_for_psv_json_stringify():
    record = {
        "input": {"Role": "A"},
        "parsed": {"extra": {"b": 1, "a": 2}},
        "meta": {},
    }

    flat = flatten_for_psv(record, input_columns=["Role"])

    # must be deterministic sorted JSON
    assert flat["extra"] == json.dumps({"a": 2, "b": 1}, ensure_ascii=False, sort_keys=True)


def test_flatten_for_psv_semantic_collision_question_id_is_prefixed():
    """
    If input has 'Question ID' and parsed has 'question_id', we must NOT overwrite.
    Parsed should become 'parsed_question_id'.
    """
    rec = {
        "input": {"Question ID": "3", "Role": "Data Scientist"},
        "parsed": {"question_id": "3", "question_name": "X"},
        "judge": None,
        "meta": {"row_index": 0},
    }

    out = flatten_for_psv(rec, input_columns=["Role", "Question ID"])

    assert out["Question ID"] == "3"
    assert out["parsed_question_id"] == "3"
    assert out["question_name"] == "X"


def test_flatten_for_psv_exact_collision_is_prefixed():
    """
    If parsed key exactly collides with an existing key in the output dict,
    it must be prefixed (parsed_*) to avoid overwriting.
    """
    rec = {
        "input": {"Role": "Data Scientist", "Question ID": "1"},
        "parsed": {"Role": "SHOULD_NOT_OVERWRITE_INPUT"},
        "judge": None,
        "meta": {"row_index": 0},
    }

    out = flatten_for_psv(rec, input_columns=["Role", "Question ID"])

    assert out["Role"] == "Data Scientist"
    assert out["parsed_Role"] == "SHOULD_NOT_OVERWRITE_INPUT"
