import csv

from modality_llm.tasks.generate_task import run as run_generate_task
from modality_llm.tasks.grammar_task import run as run_grammar_task


def test_grammar_task_file_missing_columns(tmp_path, capsys):
    # Write a CSV missing required columns
    csv_path = tmp_path / "bad.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["foo", "bar"])
        writer.writeheader()
        writer.writerow({"foo": "a", "bar": "b"})

    class Args:
        grammar_source = "file"
        examples_file = str(csv_path)
        num_examples = 0
        grammar_method = "sampling"
        grammar_language = "english"
        use_advi = True
        mcmc_samples = 1
        mcmc_chains = 1
        mcmc_cores = 1

    run_grammar_task(Args, model=None, model_name="model")
    out = capsys.readouterr().out
    assert "CSV file missing required columns" in out


def test_run_grammar_task_end_to_end(sample_modal_verbs, capsys, monkeypatch):
    # stub only the LLM back-end so calls are fast
    from argparse import Namespace

    import modality_llm.llm_cache as llm_cache
    import modality_llm.llm_utils as llm_utils

    monkeypatch.setattr(
        llm_utils,
        "make_generator",
        lambda model, pattern, num_samples: (
            lambda prompts, max_tokens: [["yes"]] * len(prompts)
        ),
    )
    monkeypatch.setattr(llm_cache.LLMCache, "get", lambda self, *a, **k: None)
    monkeypatch.setattr(llm_cache.LLMCache, "save", lambda self, *a, **k: None)

    args = Namespace(
        data_path=sample_modal_verbs,
        grammar_source="modal",
        examples_file=None,
        grammar_language="english",
        grammar_method="yesno_prob",
        num_samples=1,
        num_examples=0,
        use_advi=False,
        mcmc_samples=1,
        mcmc_chains=1,
        mcmc_cores=1,
    )
    run_grammar_task(args, model=None, model_name="model")
    out = capsys.readouterr().out
    assert "Yes/No Probability Method" in out


def test_generate_task_runs(monkeypatch, tmp_path):
    # Patch download_modal_verb_dataset and generate_grammar_examples_for_annotation
    monkeypatch.setattr(
        "modality_llm.examples.download_modal_verb_dataset", lambda path: path
    )
    monkeypatch.setattr(
        "modality_llm.examples.generate_grammar_examples_for_annotation",
        lambda **kwargs: kwargs,
    )

    class Args:
        data_path = str(tmp_path / "modal.jsonl")
        output_csv = str(tmp_path / "out.csv")
        gen_include_alternatives = False

    run_generate_task(Args)
