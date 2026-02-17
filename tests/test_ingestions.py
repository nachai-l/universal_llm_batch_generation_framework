from pathlib import Path

import pandas as pd
import pytest

from functions.core.ingestions import (
    clean_dataframe,
    read_delimited_table,
    clean_delimited_to_psv,
)


# ---------------------------------------------------------------------
# clean_dataframe
# ---------------------------------------------------------------------

def test_clean_dataframe_normalizes_cells():
    df = pd.DataFrame(
        {
            "a": [
                "  hello \n world  ",
                None,
                "NaN",
                "[None]",
                r"foo\,bar",
                r'baz\qux',
            ],
            "b": ["x\t y", "  ", "n/a", "NULL", "ok", "  done "],
        }
    )

    out = clean_dataframe(df)

    # whitespace normalization
    assert out.loc[0, "a"] == "hello world"

    # null handling
    assert out.loc[1, "a"] == ""
    assert out.loc[2, "a"] == ""
    assert out.loc[3, "a"] == ""

    # BACKSLASHES ARE PRESERVED (new correct behavior)
    assert out.loc[4, "a"] == r"foo\,bar"
    assert out.loc[5, "a"] == r"baz\qux"

    # column b
    assert out.loc[0, "b"] == "x y"
    assert out.loc[1, "b"] == ""
    assert out.loc[2, "b"] == ""
    assert out.loc[3, "b"] == ""
    assert out.loc[4, "b"] == "ok"
    assert out.loc[5, "b"] == "done"


def test_clean_dataframe_preserves_json_like_strings():
    """
    Ensure JSON-like content is not mutated (backslashes preserved).
    """
    df = pd.DataFrame(
        {
            "col": ['["A", "He said \\"ok\\""]']
        }
    )

    out = clean_dataframe(df)

    assert out.loc[0, "col"] == '["A", "He said \\"ok\\""]'


def test_clean_dataframe_flattens_newlines_and_tabs_only():
    """
    Ensure only structural whitespace is flattened,
    not normal characters.
    """
    df = pd.DataFrame(
        {
            "col": ["line1\nline2\tline3"]
        }
    )

    out = clean_dataframe(df)

    assert out.loc[0, "col"] == "line1 line2 line3"


# ---------------------------------------------------------------------
# read_delimited_table
# ---------------------------------------------------------------------

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

    # raw read keeps newline; cleaner flattens later
    assert df.loc[0, "col2"] == "line1\nline2"


def test_read_delimited_table_unknown_fmt_raises(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_delimited_table(p, fmt="bad")  # type: ignore[arg-type]


def test_read_delimited_table_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_delimited_table(tmp_path / "missing.csv", fmt="csv")


# ---------------------------------------------------------------------
# clean_delimited_to_psv
# ---------------------------------------------------------------------

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


def test_clean_delimited_to_psv_preserves_backslashes(tmp_path: Path):
    """
    Ensure backslashes are preserved in PSV output.

    NOTE:
    - With QUOTE_NONE + escapechar='\\', pandas will escape literal backslashes by doubling them.
      So a value containing C:\\tmp becomes C:\\\\tmp in the output file.
    """
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.psv"

    # Valid CSV:
    # - quotes inside quoted field -> doubled ("")
    # - backslashes are literal and should survive (but will be doubled in the PSV file)
    in_path.write_text(
        'col1\n'
        '"He said ""ok"" and path C:\\\\tmp"\n',
        encoding="utf-8",
    )

    clean_delimited_to_psv(in_path, out_path, fmt="csv")

    text = out_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Header + 1 row
    assert len(lines) == 2
    assert lines[0] == "col1"

    row = lines[1]

    # Quotes are escaped in PSV output under QUOTE_NONE + escapechar
    assert r'\"ok\"' in row

    # Backslashes are preserved but doubled by the writer
    assert r"C:\\\\tmp" in row
