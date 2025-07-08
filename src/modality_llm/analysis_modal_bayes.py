"""
Bayesian analysis for modal-verb classification.
"""

import traceback
from typing import Any, List, Optional

import altair as alt
import numpy as np
import polars as pl
import pymc as pm

from modality_llm.analysis_bayes_common import (
    run_inference,
)
from modality_llm.analysis_compare import ConfusionMatrixUncertaintyViz
from modality_llm.schema import Taxonomy, UnifiedResult
from modality_llm.utils import sanitize_model_name


def bayesian_analysis_modal(
    results: List[UnifiedResult],
    categories: List[str],
    model_name: str = "model",
    use_advi: bool = True,
    n_samples: int = 500,
    chains: int = 8,
    cores: Optional[int] = None,
    taxonomy: Optional[str] = None,
) -> Any:
    """
    Perform Bayesian analysis on modal verb classification results using a
    Dirichlet-Multinomial model for better uncertainty quantification.

    Args:
        results: List of UnifiedResult models
        categories: List of categories
        model_name: Name of the model (for plots)
        use_advi: Whether to use ADVI (faster) instead of MCMC sampling
        n_samples: Number of samples/iterations to use
        chains: Number of chains for MCMC sampling
        cores: Number of cores to use for sampling
        taxonomy: Taxonomy string to include in output filenames (required for correct naming)

    Returns:
        PyMC trace or InferenceData
    """
    print("\n=== BAYESIAN ANALYSIS FOR MODAL VERBS (DIRICHLET-MULTINOMIAL) ===\n")

    # Default taxonomy to Palmer if caller didn’t supply one
    if taxonomy is None:
        taxonomy = "palmer"

    # results are UnifiedResult models

    # Prepare data
    answer_counts = []
    expected_categories_list = []
    modal_verbs = []

    for result in results:
        try:
            # pull the distribution off the TaskResult for our taxonomy
            tax = Taxonomy(taxonomy)
            task = result.classification.get(tax)
            dist = task.english.distribution if task and task.english else {}
            counts = [dist.get(category, 0) for category in categories]
            answer_counts.append(counts)

            expected = result.grammatical.value
            if isinstance(expected, list):
                expected = expected[0]
            expected_categories_list.append(expected)

            modal_verbs.append(result.english_target or "unknown")
        except Exception as e:
            print(f"Error processing result: {e}")
            continue

    if not answer_counts:
        print("No valid results to analyze.")
        return None

    # Convert to numpy arrays
    answer_counts_np = np.array(answer_counts)
    n_examples = len(answer_counts)
    n_categories = len(categories)

    # Set random seed
    np.random.seed(42)

    # Create PyMC model with Dirichlet-Multinomial
    with pm.Model() as model:
        # Global concentration parameter (higher = more certain about categories)
        concentration = pm.Gamma("concentration", alpha=2, beta=1)

        # Base rates for each category (global tendency)
        base_rates = pm.Dirichlet("base_rates", a=np.ones(n_categories))

        # Per-example uncertainty: how much each example deviates from base rates
        # Higher values = more confidence in this specific example
        example_concentration = pm.Gamma(
            "example_concentration", alpha=2, beta=1, shape=n_examples
        )

        # Category probabilities for each example
        # Uses base_rates scaled by example-specific concentration
        theta = pm.Dirichlet(
            "theta",
            a=base_rates * example_concentration[:, None] * concentration,
            shape=(n_examples, n_categories),
        )

        # Likelihood
        pm.Multinomial(
            "y",
            n=answer_counts_np.sum(axis=1),
            p=theta,
            observed=answer_counts_np,
        )

        # Derived quantities for better interpretation
        # Entropy as uncertainty measure (higher = more uncertain)
        pm.Deterministic(
            "entropy", -pm.math.sum(theta * pm.math.log(theta + 1e-8), axis=1)
        )

        # Maximum probability (confidence in top choice)
        pm.Deterministic("max_prob", pm.math.max(theta, axis=1))

        # Run inference
        trace = run_inference(model, use_advi, n_samples, chains, cores)

    # Analyze and report results
    print("\n--- UNCERTAINTY ANALYSIS ---")
    print(f"Global concentration: {trace.posterior['concentration'].mean():.2f}")
    print("(Higher values = model is more confident overall)\n")

    # Get posterior samples
    theta_samples = np.array(trace.posterior["theta"].values)
    entropy_samples = np.array(trace.posterior["entropy"].values)
    max_prob_samples = np.array(trace.posterior["max_prob"].values)

    # 3. Now integrate computed uncertainty metrics back into each result dict
    # Compute means over chains and draws for each example index
    entropy_means = entropy_samples.mean(axis=(0, 1))  # shape: (n_examples,)
    max_prob_means = max_prob_samples.mean(axis=(0, 1))  # shape: (n_examples,)

    # If you need to export these, build a side‐table instead of mutating the model:
    augmented: list[dict] = []
    for i, r in enumerate(results):
        augmented.append(
            {
                "eid": r.eid,
                "Entropy_Mean": float(entropy_means[i]),
                "Max_Prob_Mean": float(max_prob_means[i]),
                "Top_Categories": [
                    categories[j]
                    for j in np.argsort(theta_samples[:, :, i, :].mean((0, 1)))[::-1][
                        :2
                    ]
                ],
                "Uncertainty_Level": (
                    "Very Low"
                    if max_prob_means[i] > 0.8
                    else "Low"
                    if max_prob_means[i] > 0.6
                    else "Medium"
                    if max_prob_means[i] > 0.4
                    else "High"
                ),
            }
        )

    # Create uncertainty report
    uncertainty_report = []

    for i, r in enumerate(results[:10]):
        expected = r.grammatical.value
        text = (r.english or "")[:50] + "..."
        theta_mean = theta_samples[:, :, i, :].mean(axis=(0, 1))
        entropy_mean = entropy_samples[:, :, i].mean()
        max_prob_mean = max_prob_samples[:, :, i].mean()
        top_indices = np.argsort(theta_mean)[-2:][::-1]
        top_categories = [categories[idx] for idx in top_indices]
        top_probs = [theta_mean[idx] for idx in top_indices]
        if max_prob_mean > 0.8:
            uncertainty_level = "Very Low"
        elif max_prob_mean > 0.6:
            uncertainty_level = "Low"
        elif max_prob_mean > 0.4:
            uncertainty_level = "Medium"
        else:
            uncertainty_level = "High"
        uncertainty_report.append(
            {
                "example_id": i + 1,
                "modal_verb": r.english_target or "unknown",
                "text_preview": text,
                "expected": expected,
                "top_category": top_categories[0],
                "top_prob": top_probs[0],
                "second_category": top_categories[1],
                "second_prob": top_probs[1],
                "entropy": entropy_mean,
                "max_prob": max_prob_mean,
                "uncertainty": uncertainty_level,
            }
        )

        print(f"\nExample {i + 1} - {r.english_target or 'unknown'}:")
        print(f"  Text: {text[:60]}...")
        print(f"  Expected: {expected}")
        print(f"  Top prediction: {top_categories[0]} ({top_probs[0]:.3f})")
        print(f"  Runner-up: {top_categories[1]} ({top_probs[1]:.3f})")
        print(f"  Uncertainty: {uncertainty_level} (entropy={entropy_mean:.3f})")

    # Summary statistics
    print("\n--- OVERALL UNCERTAINTY SUMMARY ---")
    all_max_probs = max_prob_samples.reshape(-1, n_examples).mean(axis=0)
    print(f"Average confidence (max probability): {all_max_probs.mean():.3f}")
    print(
        f"Examples with high uncertainty (max_prob < 0.5): {(all_max_probs < 0.5).sum()} / {n_examples}"
    )
    print(
        f"Examples with very high confidence (max_prob > 0.8): {(all_max_probs > 0.8).sum()} / {n_examples}"
    )

    # Visualizations
    sanitized_name = sanitize_model_name(model_name)

    # 1. Violin + jitter plot of max‐probability by modal verb
    df_v = pl.DataFrame(
        {
            "modal_verb": modal_verbs,
            "max_probability": all_max_probs,
        }
    )
    # build a horizontal violin + jitter: layer first, then facet by modal_verb
    density = (
        alt.Chart(df_v)
        .transform_density(
            "max_probability",
            as_=["max_probability", "density"],
            extent=[0, 1],
            groupby=["modal_verb"],
        )
        .mark_area(orient="horizontal", opacity=0.3)
        .encode(
            alt.X(
                "density:Q",
                stack="center",
                impute=None,
                title=None,
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
            ),
            alt.Y("max_probability:Q", title="Max Probability"),
            alt.Color("modal_verb:N", legend=None),
        )
    )
    points = (
        alt.Chart(df_v)
        .transform_calculate(jitter="random()")
        .mark_circle(size=15, opacity=0.4)
        .encode(
            x=alt.X("jitter:Q", axis=None, title=None),
            y=alt.Y("max_probability:Q", title="Max Probability"),
            color=alt.Color("modal_verb:N", legend=None),
            tooltip=["modal_verb", "max_probability"],
        )
    )
    # first size the layer, then facet
    layered = alt.layer(density, points).properties(width=100, height=80)
    violin_overlay = layered.facet(
        column=alt.Column(
            "modal_verb:N",
            title="Modal Verb",
            header=alt.Header(labelOrient="bottom", labelPadding=0),
        )
    ).properties(title=f"{sanitized_name} - Confidence Distribution by Modal Verb")
    violin_overlay.save(f"{sanitized_name}_{taxonomy}_confidence_by_modal_violin.html")

    # 2. Base rates visualization
    base_rates_samples = np.array(trace.posterior["base_rates"].values).reshape(
        -1, n_categories
    )
    base_rates_df = pl.DataFrame(
        {
            "category": categories,
            "mean_rate": base_rates_samples.mean(axis=0),
            "std": base_rates_samples.std(axis=0),
        }
    )

    base_chart = (
        alt.Chart(base_rates_df)
        .mark_bar()
        .encode(
            x=alt.X("category:N", sort=None, title="Category"),
            y=alt.Y("mean_rate:Q", title="Base Rate"),
            tooltip=["category", "mean_rate", "std"],
        )
        .properties(
            title=f"{sanitized_name} - Learned Base Rates for Categories",
            width=400,
            height=300,
        )
    )

    error_bars = (
        alt.Chart(base_rates_df)
        .mark_errorbar()
        .encode(
            x=alt.X("category:N", sort=None), y=alt.Y("mean_rate:Q"), yError="std:Q"
        )
    )

    (base_chart + error_bars).save(f"{sanitized_name}_{taxonomy}_modal_base_rates.html")

    # 3. Uncertainty distribution
    uncertainty_dist = pl.DataFrame(
        {
            "uncertainty": ["Very Low", "Low", "Medium", "High"],
            "count": [
                (all_max_probs > 0.8).sum(),
                ((all_max_probs > 0.6) & (all_max_probs <= 0.8)).sum(),
                ((all_max_probs > 0.4) & (all_max_probs <= 0.6)).sum(),
                (all_max_probs <= 0.4).sum(),
            ],
        }
    )
    dist_chart = (
        alt.Chart(uncertainty_dist)
        .mark_bar()
        .encode(
            x=alt.X(
                "uncertainty:N",
                sort=["Very Low", "Low", "Medium", "High"],
                title="Uncertainty Level",
            ),
            y=alt.Y("count:Q", title="Number of Examples"),
            color=alt.Color(
                "uncertainty:N",
                scale=alt.Scale(
                    domain=["Very Low", "Low", "Medium", "High"],
                    range=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"],
                ),
            ),
        )
        .properties(
            title=f"{sanitized_name} - Distribution of Classification Uncertainty",
            width=400,
            height=300,
        )
    )

    dist_chart.save(f"{sanitized_name}_{taxonomy}_modal_uncertainty_distribution.html")

    # Save uncertainty report as CSV
    report_df = pl.DataFrame(uncertainty_report)
    report_df.write_csv(f"{sanitized_name}_modal_uncertainty_report.csv")
    print(
        f"\nDetailed uncertainty report saved to: {sanitized_name}_modal_uncertainty_report.csv"
    )

    # 4. Enhanced confusion‐matrix dashboard (existing)
    try:
        viz = ConfusionMatrixUncertaintyViz(results, categories, taxonomy, model_name)
        enhanced_viz = viz.create_dashboard()
        enhanced_viz.save(f"{sanitized_name}_{taxonomy}_modal_enhanced_dashboard.html")
        print(
            f"\nEnhanced dashboard saved to: {sanitized_name}_{taxonomy}_modal_enhanced_dashboard.html"
        )
    except Exception as e:
        print(f"Error creating enhanced dashboard: {e}")
        traceback.print_exc()
        enhanced_viz = None

    # 5. Now merge everything into a single dashboard (only if step 4 succeeded)
    if enhanced_viz is not None:
        try:
            combined = alt.vconcat(
                violin_overlay.properties(title="Confidence by Modal Verb"),
                (base_chart + error_bars).properties(title="Learned Base Rates"),
                dist_chart.properties(title="Uncertainty Distribution"),
                enhanced_viz.properties(title="Confusion Matrix Dashboard"),
            ).resolve_scale(color="independent")

            combined.save(f"{sanitized_name}_{taxonomy}_modal_full_dashboard.html")
            print(
                f"\nModal full dashboard saved to: {sanitized_name}_{taxonomy}_modal_full_dashboard.html"
            )
        except Exception as e:
            print(f"Error creating modal full dashboard: {e}")
            traceback.print_exc()

    return trace
