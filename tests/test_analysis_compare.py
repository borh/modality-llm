import pytest
from modality_llm.analysis_compare import compare_with_human_annotators
from modality_llm.schema import (
    UnifiedResult,
    TaskResult,
    LanguageResult,
    Taxonomy,
    GrammarLabel,
    PALMER_CATEGORIES,
)

def make_result(dist, expected, annots, example="ex", modal="mv"):
    # turn counts into a pseudo‐answers list
    answers = []
    for k, v in dist.items():
        answers.extend([k] * v)
    lr = LanguageResult(prompt="", answers=answers)
    tr = TaskResult(english=lr)
    return UnifiedResult(
        eid=example,
        english=example,
        japanese=None,
        english_target=modal,
        japanese_target=None,
        grammatical=GrammarLabel.yes,
        grammar=None,
        human_annotations={Taxonomy.palmer: annots},
        expected_categories={Taxonomy.palmer: [expected]},
        classification={Taxonomy.palmer: tr},
    )


def test_compare_with_human_annotators_happy_path(tmp_path, capsys):
    # use actual Palmer categories
    categories = PALMER_CATEGORIES
    results = [
        make_result(
            {"deontic": 2, "epistemic": 1, "dynamic": 0, "unknown": 0},
            "deontic",
            ["deontic", "epistemic", "epistemic"],
        ),
        make_result(
            {"deontic": 0, "epistemic": 3, "dynamic": 0, "unknown": 0},
            "epistemic",
            ["epistemic", "epistemic", "epistemic"],
        ),
    ]
    compare_with_human_annotators(
        results, categories, taxonomy="palmer", model_name="testmodel"
    )
    out = capsys.readouterr().out
    assert "Confusion Matrix" in out
    assert "Consensus" in out
    assert "Agreement with consensus" in out


def test_compare_with_human_annotators_unknown_category(capsys):
    categories = PALMER_CATEGORIES
    # invalid expected label should now be caught inside compare_with_human_annotators
    results = [
        make_result(
            {"deontic": 1, "epistemic": 0, "dynamic": 0, "unknown": 0},
            "invalid",                      # not in PALMER_CATEGORIES
            ["deontic", "epistemic", "epistemic"],
        ),
    ]
    with pytest.raises(ValueError) as exc:
        compare_with_human_annotators(results, categories, "palmer", "testmodel")
    assert "Unknown consensus category" in str(exc.value)


def test_compare_with_human_annotators_fleiss_kappa_branch(tmp_path, capsys):
    categories = PALMER_CATEGORIES
    # 3 annotators + LLM, all say the same palmer category
    results = [
        make_result(
            {"dynamic": 4, "deontic": 0, "epistemic": 0, "unknown": 0},
            "dynamic",
            ["dynamic", "dynamic", "dynamic"],
        ),
        make_result(
            {"epistemic": 4, "deontic": 0, "dynamic": 0, "unknown": 0},
            "epistemic",
            ["epistemic", "epistemic", "epistemic"],
        ),
    ]
    compare_with_human_annotators(
        results, categories, taxonomy="palmer", model_name="testmodel"
    )
    out = capsys.readouterr().out
    assert "Fleiss' kappa" in out
