import numpy as np
import pymc as pm

from modality_llm.analysis_bayes_common import run_inference
from modality_llm.analysis_modal_bayes import bayesian_analysis_modal


def test_bayesian_analysis_modal_prints_and_runs(monkeypatch, capsys):
    # Use a tiny PyMC model + our real run_inference to get a legitimate trace
    with pm.Model() as m:
        # add concentration to satisfy bayesian_analysis_modal’s print
        pm.Gamma("concentration", alpha=2, beta=1)
        base = pm.Dirichlet("base_rates", a=np.ones(4))
        theta = pm.Dirichlet("theta", a=base * 1.0, shape=(1, 4))
        pm.Multinomial(
            "y", n=np.array([[2, 1, 0, 0]]).sum(), p=theta, observed=[[2, 1, 0, 0]]
        )
        pm.Deterministic("entropy", -(theta * pm.math.log(theta + 1e-8)).sum(axis=1))
        pm.Deterministic("max_prob", pm.math.max(theta, axis=1))
        real_trace = run_inference(m, use_advi=True, n_samples=1, chains=1, cores=1)

    monkeypatch.setattr(
        "modality_llm.analysis_modal_bayes.run_inference",
        lambda *a, **kw: real_trace,
        raising=True,
    )
    # build a real UnifiedResult
    from modality_llm.schema import (
        GrammarLabel,
        LanguageResult,
        TaskResult,
        Taxonomy,
        UnifiedResult,
    )

    answers = ["epistemic"] * 2 + ["deontic"]
    lr = LanguageResult(prompt="", answers=answers)
    tr = TaskResult(english=lr)
    res = UnifiedResult(
        eid="e1",
        english="You *must* go.",
        japanese=None,
        english_target="must",
        japanese_target=None,
        grammatical=GrammarLabel.yes,
        grammar=None,
        human_annotations=None,
        expected_categories={Taxonomy.palmer: ["epistemic"]},
        classification={Taxonomy.palmer: tr},
    )
    categories = ["epistemic", "deontic", "dynamic", "unknown"]
    trace = bayesian_analysis_modal(
        results=[res],
        categories=categories,
        model_name="model",
        use_advi=True,
        n_samples=1,
        chains=1,
        cores=1,
    )
    out = capsys.readouterr().out
    assert hasattr(trace, "posterior")
    assert "UNCERTAINTY ANALYSIS" in out or "OVERALL UNCERTAINTY SUMMARY" in out
