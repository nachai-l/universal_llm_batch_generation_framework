import json
from pathlib import Path

import pandas as pd

from functions.io.writers import ensure_parent_dir, write_csv, write_jsonl


def test_ensure_parent_dir_creates(tmp_path: Path):
    out = tmp_path / "a" / "b" / "file.jsonl"
    assert not out.parent.exists()
    ensure_parent_dir(out)
    assert out.parent.exists()


def test_write_jsonl_writes_deterministically(tmp_path: Path):
    out = tmp_path / "x" / "out.jsonl"
    records = [
        {"b": 2, "a": 1},
        {"z": "ไทย", "c": 3},
    ]
    write_jsonl(out, records)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    # sort_keys=True => "a" before "b"
    assert lines[0] == json.dumps({"b": 2, "a": 1}, ensure_ascii=False, sort_keys=True)

    # should preserve unicode
    obj = json.loads(lines[1])
    assert obj["z"] == "ไทย"


def test_write_csv_writes_and_creates_parent(tmp_path: Path):
    out = tmp_path / "reports" / "t.csv"
    df = pd.DataFrame({"b": [2, 3], "a": [1, 4]})

    write_csv(out, df)

    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == "b,a"  # preserves df.columns order
    assert text[1] == "2,1"
    assert text[2] == "3,4"
