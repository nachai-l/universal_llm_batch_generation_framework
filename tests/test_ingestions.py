# tests/test_ingestions.py
from pathlib import Path

import pandas as pd
import pytest

from functions.core.ingestions import (
    clean_dataframe,
    read_delimited_table,
    clean_delimited_to_psv,
)


def test_clean_dataframe_normalizes_cells():
    df = pd.DataFrame(
        {
            "a": ["  hello \n world  ", None, "NaN", "[None]", r"foo\,bar", r"baz\qux"],
            "b": ["x\t y", "  ", "n/a", "NULL", "ok", "  done "],
        }
    )

    out = clean_dataframe(df)

    assert out.loc[0, "a"] == "hello world"
    assert out.loc[1, "a"] == ""
    assert out.loc[2, "a"] == ""
    assert out.loc[3, "a"] == ""
    assert out.loc[4, "a"] == "foo,bar"
    assert out.loc[5, "a"] == "bazqux"

    assert out.loc[0, "b"] == "x y"
    assert out.loc[1, "b"] == ""
    assert out.loc[2, "b"] == ""
    assert out.loc[3, "b"] == ""
    assert out.loc[4, "b"] == "ok"
    assert out.loc[5, "b"] == "done"


def test_read_delimited_table_csv_multiline(tmp_path: Path):
    p = tmp_path / "in.csv"
    p.write_text(
        "col1,col2\n"
        'A,"line1\nline2"\n'
        "B,C\n",
        encoding="utf-8",
    )

    df = read_delimited_table(p, fmt="csv")
    assert list(df.columns) == ["col1", "col2"]
    assert df.shape[0] == 2
    assert df.loc[0, "col2"] == "line1\nline2"  # raw read keeps newline; cleaner flattens later


def test_clean_delimited_to_psv_writes_output(tmp_path: Path):
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.psv"

    in_path.write_text(
        "Role Track Example Name,Assumed Lv,Question Set,Question ID,Question Type\n"
        'Data Scientist,"Junior \n- 0 to 2 years",A,1,Generic\n',
        encoding="utf-8",
    )

    df_clean = clean_delimited_to_psv(in_path, out_path, fmt="csv")

    assert out_path.exists()
    # newline in cell should be flattened
    assert df_clean.loc[0, "Assumed Lv"] == "Junior - 0 to 2 years"

    # PSV should have pipe separator
    text = out_path.read_text(encoding="utf-8")
    assert "|" in text
    assert text.splitlines()[0].startswith("Role Track Example Name|Assumed Lv|")


def test_read_delimited_table_unknown_fmt_raises(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_delimited_table(p, fmt="bad")  # type: ignore[arg-type]


def test_read_delimited_table_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_delimited_table(tmp_path / "missing.csv", fmt="csv")
