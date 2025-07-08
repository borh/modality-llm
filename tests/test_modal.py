import pytest

from modality_llm.modal import compute_modal_results
from modality_llm.schema import Taxonomy


def test_get_categories_palmer_and_quirk():
    from modality_llm.modal import get_categories

    assert get_categories("palmer") == ["deontic", "epistemic", "dynamic", "unknown"]
    assert "ability" in get_categories("quirk")
    with pytest.raises(ValueError):
        get_categories("invalid")


def test_compute_modal_results_end_to_end(sample_modal_verbs, monkeypatch):
    # stub only our make_generator (downstream of outlines.generate/samplers)
    import modality_llm.llm_utils as llm_utils_mod

    monkeypatch.setattr(
        llm_utils_mod,
        "make_generator",
        lambda model, pattern, num_samples: (
            lambda prompts, max_tokens: [["epistemic"]] * len(prompts)
        ),
    )

    results = compute_modal_results(
        data_path=sample_modal_verbs,
        taxonomy=Taxonomy.palmer,
        num_samples=1,
        batch_size=10,
        use_cache=False,
        force_refresh=True,
        num_examples_to_sample=0,
        random_seed=0,
        model=None,
        model_name="model",
    )
    # we had 3 entries in sample; no augmentation → 3 results
    assert len(results) == 3
    # check one of them
    assert results[1].english_target == "must"
    dist = results[1].classification[Taxonomy.palmer].english.distribution
    assert dist == {"epistemic": 1}
