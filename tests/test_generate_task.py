from argparse import Namespace

from modality_llm.tasks.generate_task import run as run_generate_task


def test_run_generate_task_end_to_end(tmp_path, sample_modal_verbs, capsys):
    out_csv = tmp_path / "out.csv"
    args = Namespace(
        data_path=sample_modal_verbs,
        output_csv=str(out_csv),
        gen_include_alternatives=False,
    )
    run_generate_task(args)
    assert out_csv.exists()
    import polars as pl

    df = pl.read_csv(str(out_csv))
    assert set(df.columns) >= {
        "ID",
        "EID",
        "Original_Sentence",
        "Marked_Sentence_English",
    }
    assert len(df) == 3
    out = capsys.readouterr().out
    assert "Successfully generated" in out
