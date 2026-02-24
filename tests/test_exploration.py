# tests/test_exploration.py

from __future__ import annotations

import pandas as pd

from functions.core.exploration import analyze_missing_data_detailed, analyze_top_values_table


def test_analyze_missing_data_detailed_counts(capsys):
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": ["x", "", "Unclassified"],
        "c": ["ok", "ok", "ok"],
    })
    result = analyze_missing_data_detailed(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # one row per column

    row_a = result[result["Column"] == "a"].iloc[0]
    assert int(row_a["Null/NA"]) == 1
    assert int(row_a["Total Missing"]) == 1

    row_b = result[result["Column"] == "b"].iloc[0]
    assert int(row_b["Empty"]) == 1
    assert int(row_b["Unclassified"]) == 1
    assert int(row_b["Total Missing"]) == 2

    row_c = result[result["Column"] == "c"].iloc[0]
    assert int(row_c["Total Missing"]) == 0
    assert int(row_c["Valid"]) == 3

    captured = capsys.readouterr()
    assert "MISSING DATA ANALYSIS" in captured.out
    assert "SUMMARY" in captured.out


def test_analyze_missing_data_detailed_sorted_by_missing(capsys):
    df = pd.DataFrame({
        "clean": [1, 2, 3],
        "dirty": [None, None, None],
    })
    result = analyze_missing_data_detailed(df)
    assert result.iloc[0]["Column"] == "dirty"


def test_analyze_top_values_table_basic(capsys):
    df = pd.DataFrame({
        "color": ["red", "red", "blue", "green", "green", "green"],
        "ID": [1, 2, 3, 4, 5, 6],
    })
    results = analyze_top_values_table(df, top_n=2)

    assert isinstance(results, dict)
    assert "color" in results
    assert "ID" not in results  # excluded by default

    color = results["color"]
    assert color["unique_count"] == 3


def test_analyze_top_values_table_custom_exclude(capsys):
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
    })
    results = analyze_top_values_table(df, top_n=5, exclude_cols=["a"])
    assert "a" not in results
    assert "b" in results


def test_analyze_top_values_table_others_row(capsys):
    df = pd.DataFrame({"x": list(range(100))})
    results = analyze_top_values_table(df, top_n=5)
    # With 100 unique values and top_n=5, should have others
    top_vals = results["x"]["top_values"]
    others_rows = [r for r in top_vals if r["Rank"] == "*"]
    assert len(others_rows) == 1
