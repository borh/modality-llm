import io
import sys
import types

import numpy as np
import pymc as pm

from modality_llm.analysis_bayes_common import (
    beta_posterior_analysis,
    calculate_bayes_factor,
    interpret_bayes_factor,
    run_inference,
)


def test_interpret_bayes_factor_thresholds():
    assert interpret_bayes_factor(101) == "Extreme evidence"
    assert interpret_bayes_factor(50) == "Very strong evidence"
    assert interpret_bayes_factor(15) == "Strong evidence"
    assert interpret_bayes_factor(5) == "Moderate evidence"
    assert interpret_bayes_factor(1.5) == "Anecdotal evidence"
    assert interpret_bayes_factor(0.5) == "No evidence"


def test_calculate_bayes_factor_various():
    # expected = "yes"
    assert calculate_bayes_factor({"yes": 2, "no": 1}, "yes") == 2.0
    assert calculate_bayes_factor({"yes": 0, "no": 1}, "yes") == 0.0
    # expected = "no"
    assert calculate_bayes_factor({"yes": 2, "no": 1}, "no") == 0.5
    assert calculate_bayes_factor({"yes": 1, "no": 0}, "no") == 0.0
    # zeros, epsilon fallback
    assert calculate_bayes_factor({"yes": 0, "no": 0}, "yes") > 0
    assert calculate_bayes_factor({"yes": 0, "no": 0}, "no") > 0


def test_beta_posterior_analysis_known():
    mean, (low, high) = beta_posterior_analysis(
        {"yes": 2, "no": 3}, prior_alpha=1, prior_beta=1
    )
    assert abs(mean - 0.429) < 0.01
    assert 0 <= low <= mean <= high <= 1


def test_run_inference_advi_and_mcmc(monkeypatch):
    # Tiny model: single beta, single binomial
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        pm.Binomial("obs", n=1, p=theta, observed=[1])

    # Patch pm.fit and pm.sample to avoid long runtimes
    class DummyApprox:
        def sample(self, n, random_seed=None):
            return types.SimpleNamespace(posterior={"theta": np.array([[0.5]])})

    monkeypatch.setattr(pm, "fit", lambda n, method, random_seed: DummyApprox())
    monkeypatch.setattr(
        pm,
        "sample",
        lambda n,
        tune,
        chains,
        cores,
        return_inferencedata,
        random_seed: types.SimpleNamespace(posterior={"theta": np.array([[0.5]])}),
    )
    # Capture stdout
    out = io.StringIO()
    sys.stdout = out
    trace = run_inference(model, use_advi=True, n_samples=1, chains=1, cores=1)
    sys.stdout = sys.__stdout__
    assert hasattr(trace, "posterior")
    assert "Using ADVI" in out.getvalue()
    # Now test MCMC branch
    out = io.StringIO()
    sys.stdout = out
    trace = run_inference(model, use_advi=False, n_samples=1, chains=1, cores=1)
    sys.stdout = sys.__stdout__
    assert hasattr(trace, "posterior")
    assert "Using MCMC" in out.getvalue()
