import arviz as az
import pandas as pd
import pytest

from modality_llm.analysis_grammar_bayes import (
    bayesian_analysis,
    bayesian_analysis_hierarchical_grammar,
    bayesian_analysis_with_pymc,
)
from modality_llm.schema import (
    UnifiedResult,
    TaskResult,
    LanguageResult,
    GrammarLabel,
)


def test_bayesian_analysis_prints_bayes_and_kl(capsys):
    # build a fake UnifiedResult with a grammar TaskResult
    eng_ans = ["yes"] * 2 + ["no"] * 1
    jap_ans = ["yes"] * 1 + ["no"] * 2
    lr_eng = LanguageResult(prompt="", answers=eng_ans)
    lr_jap = LanguageResult(prompt="", answers=jap_ans)
    tr = TaskResult(english=lr_eng, japanese=lr_jap)
    res = UnifiedResult(
        eid="e1",
        english="ex1",
        japanese=None,
        english_target="dummy",  # required when english is set
        japanese_target=None,
        grammatical=GrammarLabel.yes,
        grammar=tr,
        human_annotations=None,
        expected_categories=None,
        classification=None,
    )
    bayesian_analysis([res])
    out = capsys.readouterr().out
    assert "Bayes factor" in out
    assert "KL Divergence" in out


def test_bayesian_analysis_with_pymc_runs(monkeypatch, capsys):
    # Patch run_inference to return a real InferenceData

    # Build minimal posterior dict for eng_theta & jap_theta
    idata = az.from_dict(
        posterior={
            "eng_theta": [[0.5]],
            "jap_theta": [[0.5]],
        }
    )
    monkeypatch.setattr(
        "modality_llm.analysis_grammar_bayes.run_inference",
        lambda *a, **kw: idata,
        raising=True,
    )
    # stub ArviZ so summary.loc['eng_theta[0]'] etc. won’t KeyError

    df = pd.DataFrame(
        {
            "mean": [0.5, 0.5],
            "hdi_3%": [0.1, 0.1],
            "hdi_97%": [0.9, 0.9],
        },
        index=["eng_theta[0]", "jap_theta[0]"],
    )
    monkeypatch.setattr(az, "ess", lambda trace, var_names: df, raising=True)
    monkeypatch.setattr(az, "rhat", lambda trace, var_names: df, raising=True)
    monkeypatch.setattr(az, "summary", lambda trace, var_names: df, raising=True)
    # build a fake UnifiedResult for pymc path
    eng_ans = ["yes"]
    jap_ans = ["no"]
    lr_eng = LanguageResult(prompt="", answers=eng_ans)
    lr_jap = LanguageResult(prompt="", answers=jap_ans)
    tr = TaskResult(english=lr_eng, japanese=lr_jap)
    res = UnifiedResult(
        eid="e2",
        english="ex2",
        japanese=None,
        english_target="dummy",  # required for UnifiedResult
        japanese_target=None,
        grammatical=GrammarLabel.yes,
        grammar=tr,
        human_annotations=None,
        expected_categories=None,
        classification=None,
    )
    trace = bayesian_analysis_with_pymc(
        [res], model_name="model", use_advi=True, n_samples=1, chains=1, cores=1
    )
    out = capsys.readouterr().out
    assert hasattr(trace, "posterior")
    assert "Convergence Diagnostics" in out


def test_bayesian_analysis_hierarchical_grammar_warns_on_missing_keys(capsys):
    # Missing EID and english_target → should trigger the warning branch
    # Provide an empty LanguageResult so require_some_output passes
    from modality_llm.schema import LanguageResult

    tr = TaskResult(english=LanguageResult(prompt="", answers=[]), japanese=None)
    res = UnifiedResult(
        eid="",
        english=None,
        japanese=None,
        english_target=None,
        japanese_target=None,
        grammatical=GrammarLabel.yes,
        grammar=tr,
        human_annotations=None,
        expected_categories=None,
        classification=None,
    )
    bayesian_analysis_hierarchical_grammar(
        [res], language="English", use_advi=True, n_samples=1
    )
    out = capsys.readouterr().out
    assert "Warning: Results lack EID" in out or "missing 'EID'" in out


def test_bayesian_analysis_hierarchical_grammar_happy(monkeypatch):
    # Patch run_inference to return the same minimal InferenceData

    idata = az.from_dict(
        posterior={
            "base_intercept": [0.0],
            "sigma_sentence": [1.0],
            "sigma_modal": [1.0],
            "sentence_effect": [[0.0]],
            "modal_effect": [[0.0]],
        }
    )
    monkeypatch.setattr(
        "modality_llm.analysis_grammar_bayes.run_inference",
        lambda *a, **kw: idata,
        raising=True,
    )
    # Provide EID and english_target so the hierarchical path runs
    eng_ans = ["yes"]
    tr = TaskResult(english=LanguageResult(prompt="", answers=eng_ans), japanese=None)
    res = UnifiedResult(
        eid="e1",
        english="ex3",
        japanese=None,
        english_target="must",
        japanese_target=None,
        grammatical=GrammarLabel.yes,
        grammar=tr,
        human_annotations=None,
        expected_categories=None,
        classification=None,
    )
    trace = bayesian_analysis_hierarchical_grammar(
        [res], language="English", use_advi=True, n_samples=1
    )
    assert hasattr(trace, "posterior")
