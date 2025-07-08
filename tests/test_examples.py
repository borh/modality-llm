import os

import polars as pl

from modality_llm.examples import (
    download_modal_verb_dataset,
    generate_grammar_examples_for_annotation,
)


def test_download_modal_verb_dataset_failure(tmp_path, monkeypatch):
    # Point to a non-writable directory
    unwritable = tmp_path / "nope" / "file.jsonl"
    # Patch requests.get to raise
    import requests

    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(
            requests.exceptions.RequestException("fail")
        ),
    )
    ret = download_modal_verb_dataset(str(unwritable))
    # Should return the path string even on failure
    assert ret == str(unwritable)
    assert os.path.exists(unwritable)
    # Verify file is empty (0 bytes)
    assert os.path.getsize(unwritable) == 0


def test_generate_grammar_examples_for_annotation_end_to_end(tmp_path):
    # Write a fake JSONL
    data = [
        {"mv": "can", "utt": "I *can* swim.", "res": {}, "annotations": {}},
        {"mv": "must", "utt": "You *must* go.", "res": {}, "annotations": {}},
    ]
    jl = tmp_path / "data.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(str(rec).replace("'", '"') + "\n")
    out_csv = tmp_path / "out.csv"
    examples = generate_grammar_examples_for_annotation(
        str(jl), str(out_csv), include_alternatives=False
    )
    assert examples
    df = pl.read_csv(str(out_csv))
    assert set(df.columns) >= {
        "ID",
        "EID",
        "Original_Sentence",
        "Marked_Sentence_English",
    }
    assert len(df) == 2
    assert (
        "can" in df["Marked_Sentence_English"][0]
        or "must" in df["Marked_Sentence_English"][1]
    )
