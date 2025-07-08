from modality_llm.tasks.modal_task import run as run_modal_task


def test_run_modal_task_end_to_end(sample_modal_verbs, capsys, monkeypatch):
    # stub only make_generator + disable caching
    import modality_llm.llm_cache as llm_cache_mod
    import modality_llm.llm_utils as llm_utils_mod

    monkeypatch.setattr(
        llm_utils_mod,
        "make_generator",
        lambda model, pattern, num_samples: (
            lambda prompts, max_tokens: [["epistemic"]] * len(prompts)
        ),
    )
    monkeypatch.setattr(llm_cache_mod.LLMCache, "get", lambda self, *a, **k: None)
    monkeypatch.setattr(llm_cache_mod.LLMCache, "save", lambda self, *a, **k: None)

    class Args:
        data_path = sample_modal_verbs
        taxonomy = "both"
        num_samples = 1
        batch_size = 2
        use_cache = False
        force_refresh = True
        num_examples = 0
        random_seed = 0

    # run both palmer & quirk
    run_modal_task(Args, model=None, model_name="model")
    out = capsys.readouterr().out
    assert "Full Palmer results written" in out
    assert "Full Quirk results written" in out
