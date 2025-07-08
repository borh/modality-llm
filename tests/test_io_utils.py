import csv
import json

from modality_llm.utils import load_csv, load_jsonl


def test_load_jsonl_and_load_csv(tmp_path):
    # JSONL
    data = [{"x": 1}, {"y": 2}]
    jl = tmp_path / "data.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec) + "\n")
    assert load_jsonl(str(jl)) == data

    # CSV
    rows = [{"a": "foo", "b": "bar"}, {"a": "baz", "b": "qux"}]
    cf = tmp_path / "data.csv"
    with open(cf, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["a", "b"])
        writer.writeheader()
        writer.writerows(rows)
    assert load_csv(str(cf)) == rows
