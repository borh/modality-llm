"""
Bayesian analyses for grammar‐checking results.
"""

from typing import Any, List, Literal, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from pydantic import BaseModel

from modality_llm.analysis_bayes_common import (
    beta_posterior_analysis,
    calculate_bayes_factor,
    interpret_bayes_factor,
    run_inference,
)
from modality_llm.schema import GrammarLabel, UnifiedResult
from modality_llm.utils import sanitize_model_name


class GrammarAnalysisResults(BaseModel):
    """Pure result of grammar Bayesian analysis."""

    english_posterior: Optional[dict[str, float]] = None
    japanese_posterior: Optional[dict[str, float]] = None
    kl_divergence: Optional[float] = None
    # optional PyMC‐based outputs
    trace: Optional[Any] = None
    summary: Optional[Any] = None
    # optional hierarchical outputs
    hier_trace: Optional[Any] = None
    hier_summary: Optional[Any] = None
    plot_prefix: Optional[str] = None


def compute_grammar_analysis(
    results: List[UnifiedResult],
) -> GrammarAnalysisResults:
    """
    Pure computation of flat Bayesian analysis.
    """
    results_list = results
    english_prior = {"yes": 0.5, "no": 0.5}
    japanese_prior = {"yes": 0.5, "no": 0.5}

    for result in results_list:
        # pull distributions off the grammar.TaskResult
        eng_dist = result.grammar.english.distribution
        jap_dist = (
            result.grammar.japanese.distribution
            if result.grammar.japanese
            else {"yes": 0, "no": 0}
        )

        eng_post_mean, _ = beta_posterior_analysis(
            eng_dist, english_prior["yes"] * 10, english_prior["no"] * 10
        )
        jap_post_mean, _ = beta_posterior_analysis(
            jap_dist, japanese_prior["yes"] * 10, japanese_prior["no"] * 10
        )
        english_prior = {"yes": eng_post_mean, "no": 1 - eng_post_mean}
        japanese_prior = {"yes": jap_post_mean, "no": 1 - jap_post_mean}

    epsilon = 1e-9
    p_eng_yes = english_prior["yes"] + epsilon
    p_eng_no = english_prior["no"] + epsilon
    p_jap_yes = japanese_prior["yes"] + epsilon
    p_jap_no = japanese_prior["no"] + epsilon

    norm_eng_yes = p_eng_yes / (p_eng_yes + p_eng_no)
    norm_eng_no = p_eng_no / (p_eng_yes + p_eng_no)
    norm_jap_yes = p_jap_yes / (p_jap_yes + p_jap_no)
    norm_jap_no = p_jap_no / (p_jap_yes + p_jap_no)

    kl_div = norm_eng_yes * np.log(norm_eng_yes / norm_jap_yes) + norm_eng_no * np.log(
        norm_eng_no / norm_jap_no
    )

    return GrammarAnalysisResults(
        english_posterior=english_prior,
        japanese_posterior=japanese_prior,
        kl_divergence=kl_div,
    )


def report_grammar_analysis(
    ar: GrammarAnalysisResults, results: List[UnifiedResult]
) -> None:
    """
    Side-effecting report: prints summaries for flat Bayesian analysis.
    """
    print("\n=== BAYESIAN ANALYSIS ===\n")
    english_prior = ar.english_posterior
    japanese_prior = ar.japanese_posterior
    kl_div = ar.kl_divergence
    # narrow away Optional[...] so mypy knows these are real dicts/floats
    if english_prior is None or japanese_prior is None or kl_div is None:
        raise ValueError("Missing flat‐analysis results for reporting")

    # Re-run the per-example printout for the first 3 examples
    for i, result in enumerate(results[:3]):
        # expected grammaticality is stored in .grammatical
        expected = result.grammatical.value
        # pull the yes/no counts off the grammar TaskResult
        eng_dist = result.grammar.english.distribution
        jap_dist = (
            result.grammar.japanese.distribution
            if result.grammar.japanese
            else {"yes": 0, "no": 0}
        )
        eng_bayes_factor = calculate_bayes_factor(eng_dist, expected)
        jap_bayes_factor = calculate_bayes_factor(jap_dist, expected)
        print(f"\nExample {i + 1}:")
        print(f"  Expected answer: {expected}")
        print(
            f"  English Bayes factor: {eng_bayes_factor:.2f} - {interpret_bayes_factor(eng_bayes_factor)}"
        )
        print(
            f"  Japanese Bayes factor: {jap_bayes_factor:.2f} - {interpret_bayes_factor(jap_bayes_factor)}"
        )
        eng_post_mean, eng_interval = beta_posterior_analysis(eng_dist, 1, 1)
        jap_post_mean, jap_interval = beta_posterior_analysis(jap_dist, 1, 1)
        print(
            f"  English posterior probability of 'yes': {eng_post_mean:.2f} (95% CI: {eng_interval[0]:.2f}-{eng_interval[1]:.2f})"
        )
        print(
            f"  Japanese posterior probability of 'yes': {jap_post_mean:.2f} (95% CI: {jap_interval[0]:.2f}-{jap_interval[1]:.2f})"
        )
        print()

    print("Overall Language Model Comparison:")
    print(
        f"  English final posterior: P(yes)={english_prior['yes']:.2f}, P(no)={english_prior['no']:.2f}"
    )
    print(
        f"  Japanese final posterior: P(yes)={japanese_prior['yes']:.2f}, P(no)={japanese_prior['no']:.2f}"
    )
    print(f"  KL Divergence between English and Japanese distributions: {kl_div:.4f}")
    if kl_div < 0.05:
        print("  The model shows similar behavior across both languages.")
    else:
        print("  The model shows different behavior between languages.")


def compute_grammar_analysis_with_pymc(
    results: List[UnifiedResult],
    model_name: str = "model",
    use_advi: bool = True,
    n_samples: int = 500,
    chains: int = 8,
    cores: Optional[int] = None,
) -> GrammarAnalysisResults:
    """
    Pure computation of PyMC-based Bayesian analysis.
    """
    # work straight on Pydantic UnifiedResult
    results_to_process = results

    eng_yes_counts = []
    eng_no_counts = []
    jap_yes_counts = []
    jap_no_counts = []
    expected_values = []

    for result in results_to_process:
        eng_dist = result.grammar.english.distribution
        jap_dist = (
            result.grammar.japanese.distribution
            if result.grammar.japanese
            else {"yes": 0, "no": 0}
        )
        # expected grammaticality is stored in result.grammatical
        exp_val = 1 if result.grammatical == GrammarLabel.yes else 0
        eng_yes_counts.append(eng_dist.get("yes", 0))
        eng_no_counts.append(eng_dist.get("no", 0))
        jap_yes_counts.append(jap_dist.get("yes", 0))
        jap_no_counts.append(jap_dist.get("no", 0))
        expected_values.append(exp_val)

    np.random.seed(42)

    with pm.Model() as model:  # type: ignore[context-manager]  # noqa: F841
        eng_theta = pm.Beta("eng_theta", alpha=1, beta=1, shape=len(results_to_process))
        jap_theta = pm.Beta("jap_theta", alpha=1, beta=1, shape=len(results_to_process))
        eng_yes = pm.Binomial(  # noqa: F841
            "eng_yes",
            n=np.array(eng_yes_counts) + np.array(eng_no_counts),
            p=eng_theta,
            observed=eng_yes_counts,
        )
        jap_yes = pm.Binomial(  # noqa: F841
            "jap_yes",
            n=np.array(jap_yes_counts) + np.array(jap_no_counts),
            p=jap_theta,
            observed=jap_yes_counts,
        )
        trace = run_inference(model, use_advi, n_samples, chains, cores)

    summary: Any = az.summary(trace, var_names=["eng_theta", "jap_theta"])
    return GrammarAnalysisResults(trace=trace, summary=summary)


def report_grammar_analysis_with_pymc(
    ar: GrammarAnalysisResults,
    results: List[UnifiedResult],
    model_name: str = "model",
) -> None:
    """
    Side-effecting report: prints summaries for PyMC-based Bayesian analysis.
    """
    print("\n=== BAYESIAN ANALYSIS WITH PYMC ===\n")
    trace = ar.trace
    summary = ar.summary
    # narrow away Optional[Any] so we can call .loc without error
    if trace is None or summary is None:
        raise ValueError("Missing PyMC‐analysis results for reporting")

    print("\nConvergence Diagnostics:")
    ess_df = az.ess(trace, var_names=["eng_theta", "jap_theta"])
    rhat_df = az.rhat(trace, var_names=["eng_theta", "jap_theta"])
    print("Effective Sample Size (ESS):")
    print(ess_df)
    print("\nR-hat values (should be close to 1.0):")
    print(rhat_df)

    print("\nStatistical Summary:")
    print(summary)

    for i, result in enumerate(results):
        eng_mean = summary.loc[f"eng_theta[{i}]", "mean"]
        eng_hdi = (
            summary.loc[f"eng_theta[{i}]", "hdi_3%"],
            summary.loc[f"eng_theta[{i}]", "hdi_97%"],
        )
        jap_mean = summary.loc[f"jap_theta[{i}]", "mean"]
        jap_hdi = (
            summary.loc[f"jap_theta[{i}]", "hdi_3%"],
            summary.loc[f"jap_theta[{i}]", "hdi_97%"],
        )
        expected = result.grammatical.value
        print(f"\nExample {i + 1}:")
        print(f"  Expected answer: {expected}")
        print(
            f"  English probability of 'yes': {eng_mean:.2f} (94% HDI: {eng_hdi[0]:.2f}-{eng_hdi[1]:.2f})"
        )
        print(
            f"  Japanese probability of 'yes': {jap_mean:.2f} (94% HDI: {jap_hdi[0]:.2f}-{jap_hdi[1]:.2f})"
        )
        expected_val = 1 if expected == "yes" else 0
        eng_bf = (
            eng_mean / (1 - eng_mean)
            if expected_val == 1
            else (1 - eng_mean) / eng_mean
        )
        jap_bf = (
            jap_mean / (1 - jap_mean)
            if expected_val == 1
            else (1 - jap_mean) / jap_mean
        )
        print(
            f"  English Bayes factor (approx): {eng_bf:.2f} - {interpret_bayes_factor(eng_bf)}"
        )
        print(
            f"  Japanese Bayes factor (approx): {jap_bf:.2f} - {interpret_bayes_factor(jap_bf)}"
        )


def compute_grammar_analysis_hierarchical(
    results: List[UnifiedResult],
    language: Literal["English", "Japanese"],
    use_advi: bool = True,
    n_samples: int = 500,
    chains: int = 8,
    cores: Optional[int] = None,
) -> GrammarAnalysisResults:
    """
    Pure computation of hierarchical Bayesian analysis.
    """
    results_list = results
    eids = []
    modals = []
    yes_counts = []
    total_counts = []

    # We only do hierarchical if we have EIDs and english_target (tested_modal)
    has_grouping_info = bool(
        results_list and results_list[0].eid and results_list[0].english_target
    )

    if not has_grouping_info:
        return GrammarAnalysisResults(hier_trace=None, hier_summary=None)

    for res in results_list:
        try:
            dist = (
                res.grammar.english.distribution
                if language == "English"
                else res.grammar.japanese.distribution
            )
            yes = dist.get("yes", 0)
            no = dist.get("no", 0)
            total = yes + no
            if total > 0:
                eid = res.eid
                modal = res.english_target or res.japanese_target or "unknown"
                eids.append(eid)
                modals.append(modal)
                yes_counts.append(yes)
                total_counts.append(total)
        except (KeyError, TypeError):
            continue

    if not yes_counts:
        return GrammarAnalysisResults(hier_trace=None, hier_summary=None)

    unique_eids = sorted(list(set(eids)))
    unique_modals = sorted(list(set(modals)))
    eid_map = {eid: i for i, eid in enumerate(unique_eids)}
    modal_map = {modal: i for i, modal in enumerate(unique_modals)}

    eid_indices = [eid_map[eid] for eid in eids]
    modal_indices = [modal_map[modal] for modal in modals]

    yes_counts_np = np.array(yes_counts)
    total_counts_np = np.array(total_counts)
    eid_indices_np = np.array(eid_indices)
    modal_indices_np = np.array(modal_indices)

    coords = {
        "eid": unique_eids,
        "modal": unique_modals,
        "obs_id": np.arange(len(yes_counts_np)),
    }

    with pm.Model(coords=coords) as model:  # type: ignore[context-manager]
        sigma_sentence = pm.HalfNormal("sigma_sentence", sigma=1.0)
        sigma_modal = pm.HalfNormal("sigma_modal", sigma=1.0)
        base_intercept = pm.Normal("base_intercept", mu=0, sigma=1.5)
        sentence_effect_raw = pm.Normal(
            "sentence_effect_raw", mu=0, sigma=1, dims="eid"
        )  # type: ignore[call-arg]
        sentence_effect = pm.Deterministic(
            "sentence_effect", sentence_effect_raw * sigma_sentence, dims="eid"
        )  # type: ignore[call-arg]
        modal_effect_raw = pm.Normal("modal_effect_raw", mu=0, sigma=1, dims="modal")  # type: ignore[call-arg]
        modal_effect = pm.Deterministic(
            "modal_effect", modal_effect_raw * sigma_modal, dims="modal"
        )  # type: ignore[call-arg]
        logit_p = (
            base_intercept
            + sentence_effect[eid_indices_np]
            + modal_effect[modal_indices_np]
        )
        p = pm.Deterministic("p", pm.invlogit(logit_p), dims="obs_id")
        _ = pm.Binomial(
            "y_obs", n=total_counts_np, p=p, observed=yes_counts_np, dims="obs_id"
        )
        trace = run_inference(model, use_advi, n_samples, chains, cores)

    summary = az.summary(
        trace,
        var_names=[
            "base_intercept",
            "sigma_sentence",
            "sigma_modal",
            "sentence_effect",
            "modal_effect",
        ],
        hdi_prob=0.94,
    )
    return GrammarAnalysisResults(
        hier_trace=trace, hier_summary=summary, plot_prefix="model"
    )


def report_grammar_analysis_hierarchical(
    ar: GrammarAnalysisResults,
    language: Literal["English", "Japanese"],
) -> None:
    """
    Side-effecting report: prints summaries and plots for hierarchical Bayesian analysis.
    """
    print(f"\n=== HIERARCHICAL BAYESIAN ANALYSIS ({language}) ===\n")
    trace = ar.hier_trace
    summary = ar.hier_summary
    if trace is None or summary is None:
        print(
            "Warning: Results lack EID or Tested_Modal keys. Cannot perform hierarchical analysis by sentence/modal."
        )
        return

    print("\n--- Hierarchical Model Summary ---")
    print(summary)

    try:
        sanitized_name = sanitize_model_name(ar.plot_prefix or "model")
        plot_filename_base = f"{sanitized_name}_hierarchical_{language}"

        az.plot_forest(
            trace, var_names=["sentence_effect"], combined=True, hdi_prob=0.94
        )
        plt.suptitle(f"{language} Sentence Effects (Log-Odds Deviation from Base)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f"{plot_filename_base}_sentence_effects.png")
        plt.close()

        az.plot_forest(trace, var_names=["modal_effect"], combined=True, hdi_prob=0.94)
        plt.suptitle(f"{language} Modal Effects (Log-Odds Deviation from Base)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f"{plot_filename_base}_modal_effects.png")
        plt.close()

        az.plot_posterior(
            trace, var_names=["sigma_sentence", "sigma_modal"], hdi_prob=0.94
        )
        plt.suptitle(f"{language} Variance Components (Sentence vs Modal Effects)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f"{plot_filename_base}_variance_components.png")
        plt.close()

        print(f"\nPlots saved with prefix: {plot_filename_base}_*.png")

    except Exception as e:
        print(f"Error generating plots: {e}")


def bayesian_analysis(results: List[UnifiedResult]) -> None:
    ar = compute_grammar_analysis(results)
    report_grammar_analysis(ar, results)


def bayesian_analysis_with_pymc(
    results: List[UnifiedResult],
    model_name: str = "model",
    use_advi: bool = True,
    n_samples: int = 500,
    chains: int = 8,
    cores: Optional[int] = None,
) -> Any:
    ar = compute_grammar_analysis_with_pymc(
        results, model_name, use_advi, n_samples, chains, cores
    )
    report_grammar_analysis_with_pymc(ar, results, model_name)
    return ar.trace


def bayesian_analysis_hierarchical_grammar(
    results: List[UnifiedResult],
    language: Literal["English", "Japanese"],
    use_advi: bool = True,
    n_samples: int = 500,
    chains: int = 8,
    cores: Optional[int] = None,
) -> Optional[az.InferenceData]:
    ar = compute_grammar_analysis_hierarchical(
        results, language, use_advi, n_samples, chains, cores
    )
    report_grammar_analysis_hierarchical(ar, language)
    return ar.hier_trace
