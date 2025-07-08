"""
Compare LLM predictions with human annotators.
"""

import traceback
from typing import List, Literal

import altair as alt
import numpy as np
import polars as pl
from statsmodels.stats.inter_rater import fleiss_kappa

from modality_llm.schema import Taxonomy, UnifiedResult
from modality_llm.utils import sanitize_model_name


# Enhanced visualization class for confusion matrix uncertainty
class ConfusionMatrixUncertaintyViz:
    """Enhanced confusion matrix visualization with uncertainty quantification."""

    def __init__(
        self,
        results: List[UnifiedResult],
        categories: List[str],
        taxonomy: str,
        model_name: str,
    ):
        """Initialize with results instead of confusion matrices."""
        self.results = results
        self.categories = categories
        self.taxonomy = taxonomy
        self.model_name = model_name
        self.n_classes = len(categories)
        self.prepare_confusion_data()

    def prepare_confusion_data(self):
        """Extract confusion data from results with uncertainty via bootstrapping."""
        n_bootstrap = 100
        confusion_matrices = []

        tax_enum = Taxonomy(self.taxonomy)
        for _ in range(n_bootstrap):
            idxs = np.random.choice(
                len(self.results), size=len(self.results), replace=True
            )
            sampled_results = [self.results[i] for i in idxs]
            cm = np.zeros((self.n_classes, self.n_classes))
            for result in sampled_results:
                # human consensus label
                exp_list = (result.expected_categories or {}).get(tax_enum, [])
                true_label = exp_list[0] if exp_list else "unknown"
                # LLM prediction
                task = result.classification.get(tax_enum)
                dist = task.english.distribution if task and task.english else {}
                pred_label = (
                    max(dist.items(), key=lambda x: x[1])[0] if dist else "unknown"
                )
                true_idx = (
                    self.categories.index(true_label)
                    if true_label in self.categories
                    else -1
                )
                pred_idx = (
                    self.categories.index(pred_label)
                    if pred_label in self.categories
                    else -1
                )
                if true_idx >= 0 and pred_idx >= 0:
                    cm[true_idx, pred_idx] += 1
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_normalized = np.divide(cm, row_sums, where=row_sums > 0)
            confusion_matrices.append(cm_normalized)
        self.confusion_matrices = np.array(confusion_matrices)
        self.prepare_data()

    def prepare_data(self):
        # Compute mean, std, and 95% CI for each cell
        self.mean_matrix = np.mean(self.confusion_matrices, axis=0)
        self.std_matrix = np.std(self.confusion_matrices, axis=0)
        self.lower_95 = np.percentile(self.confusion_matrices, 2.5, axis=0)
        self.upper_95 = np.percentile(self.confusion_matrices, 97.5, axis=0)
        # Prepare dataframe for Altair
        rows = []
        for i, true_cat in enumerate(self.categories):
            for j, pred_cat in enumerate(self.categories):
                rows.append(
                    {
                        "expected": true_cat,
                        "predicted": pred_cat,
                        "mean": self.mean_matrix[i, j],
                        "std": self.std_matrix[i, j],
                        "lower_95": self.lower_95[i, j],
                        "upper_95": self.upper_95[i, j],
                    }
                )
        self.df = pl.DataFrame(rows)

    def create_main_heatmap(self):
        # Main heatmap with mean and error bars
        base = (
            alt.Chart(self.df)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", title="LLM Prediction"),
                y=alt.Y("expected:N", title="Human Consensus"),
                color=alt.Color(
                    "mean:Q", scale=alt.Scale(scheme="blues"), title="Mean Rate"
                ),
                tooltip=[
                    "expected",
                    "predicted",
                    "mean",
                    "std",
                    "lower_95",
                    "upper_95",
                ],
            )
        )
        text = (
            alt.Chart(self.df)
            .mark_text(baseline="middle")
            .encode(
                x="predicted:N",
                y="expected:N",
                text=alt.Text("mean:Q", format=".2f"),
                color=alt.condition(
                    "datum.mean > 0.5", alt.value("white"), alt.value("black")
                ),
            )
        )
        return (base + text).properties(
            title=f"{self.model_name} - Confusion Matrix (Mean Rates, Bootstrapped) - {self.taxonomy.upper()}",
            width=400,
            height=400,
        )

    def create_uncertainty_heatmap(self):
        # Heatmap of standard deviation (uncertainty)
        base = (
            alt.Chart(self.df)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", title="LLM Prediction"),
                y=alt.Y("expected:N", title="Human Consensus"),
                color=alt.Color(
                    "std:Q", scale=alt.Scale(scheme="oranges"), title="Std Dev"
                ),
                tooltip=[
                    "expected",
                    "predicted",
                    "mean",
                    "std",
                    "lower_95",
                    "upper_95",
                ],
            )
        )
        text = (
            alt.Chart(self.df)
            .mark_text(baseline="middle")
            .encode(
                x="predicted:N",
                y="expected:N",
                text=alt.Text("std:Q", format=".2f"),
                color=alt.condition(
                    "datum.std > 0.2", alt.value("white"), alt.value("black")
                ),
            )
        )
        return (base + text).properties(
            title=f"{self.model_name} - Confusion Matrix Uncertainty (Std Dev) - {self.taxonomy.upper()}",
            width=400,
            height=400,
        )

    def create_interval_plot(self):
        """
        Show 95% CI intervals for each cell’s mean rate,
        faceted by predicted category.
        """
        # layer point marks and errorbars
        point = (
            alt.Chart(self.df)
            .mark_point(filled=True, size=60)
            .encode(
                x=alt.X("mean:Q", title="Mean Rate"),
                y=alt.Y("expected:N", title="Human Consensus"),
                color=alt.Color(
                    "mean:Q", scale=alt.Scale(scheme="blues"), title="Mean Rate"
                ),
                tooltip=[
                    "expected",
                    "predicted",
                    "mean",
                    "std",
                    "lower_95",
                    "upper_95",
                ],
            )
        )
        errorbars = (
            alt.Chart(self.df)
            .mark_errorbar()
            .encode(
                x=alt.X("mean:Q", title="Mean Rate"),
                xError="upper_95:Q",
                xError2="lower_95:Q",
                y=alt.Y("expected:N", title="Human Consensus"),
            )
        )

        # facet the layered chart by predicted category
        chart = (
            alt.layer(point, errorbars)
            .facet(column=alt.Column("predicted:N", title="LLM Prediction"))
            .properties(
                title=f"{self.model_name} - Confusion Matrix 95% Intervals - {self.taxonomy.upper()}",
                width=150,
                height=150,
            )
        )
        return chart

    def create_dashboard(self):
        """
        Combine main heatmap and uncertainty heatmap into a dashboard.

        Note: we omit the faceted interval‐plot here to avoid embedding a
        top‐level `facet` inside a vconcat (which Vega-Lite disallows).
        """
        # 1) raw counts  2) mean‐rate heatmap  3) std‐dev heatmap
        counts = self.create_count_heatmap()
        heatmap = self.create_main_heatmap()
        uncertainty = self.create_uncertainty_heatmap()
        return (
            alt.vconcat(counts, heatmap, uncertainty)
            .resolve_scale(color="independent")
            .properties(
                title=f"{self.model_name} - Confusion Matrix Overview - {self.taxonomy.upper()}"
            )
        )

    def create_count_heatmap(self):
        """
        Plot the raw confusion‐matrix counts.
        """
        # aggregate actual counts from self.results
        count_dict: dict[tuple[str, str], int] = {}
        tax_enum = Taxonomy(self.taxonomy)
        for r in self.results:
            exp_list = (r.expected_categories or {}).get(tax_enum, [])
            exp = exp_list[0] if exp_list else "unknown"
            task = r.classification.get(tax_enum)
            dist = task.english.distribution if task and task.english else {}
            pred = max(dist.items(), key=lambda x: x[1])[0] if dist else "unknown"
            count_dict[(exp, pred)] = count_dict.get((exp, pred), 0) + 1

        rows = []
        for exp in self.categories:
            for pred in self.categories:
                rows.append(
                    {
                        "expected": exp,
                        "predicted": pred,
                        "count": count_dict.get((exp, pred), 0),
                    }
                )
        df_counts = pl.DataFrame(rows)
        # contrast threshold for text color (cast to native Python float)
        max_count = float(
            max((row["count"] for row in df_counts.to_dicts()), default=0.0)
        )

        # base heatmap
        rect = (
            alt.Chart(df_counts)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", title="LLM Prediction"),
                y=alt.Y("expected:N", title="Human Consensus"),
                color=alt.Color(
                    "count:Q", scale=alt.Scale(scheme="greens"), title="Count"
                ),
                tooltip=["expected", "predicted", "count"],
            )
        )

        # centred count labels
        text = (
            alt.Chart(df_counts)
            .mark_text(baseline="middle")
            .encode(
                x="predicted:N",
                y="expected:N",
                text=alt.Text("count:Q"),
                color=alt.condition(
                    alt.datum.count > max_count / 2,
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
        )

        return (rect + text).properties(
            title=f"{self.model_name} - Raw Confusion Counts - {self.taxonomy.upper()}",
            width=400,
            height=400,
        )


def compare_with_human_annotators(
    results: List[UnifiedResult],
    categories: List[str],
    taxonomy: Literal["palmer", "quirk"],
    model_name: str = "model",
) -> None:
    """
    Compare LLM results with human annotations.

    Args:
        results: List of UnifiedResult models.
        categories: Sorted list of categories including 'unknown'.
        taxonomy: 'palmer' or 'quirk' (controls title/labels).
        model_name: Name of the model (for printing and filenames).
    """
    print(f"\n=== COMPARING LLM WITH HUMAN ANNOTATORS ({taxonomy.upper()}) ===\n")

    # We now assume results are UnifiedResult models

    # Initialize counters
    total_examples = len(results)
    agreement_with_consensus = 0
    agreement_with_any = 0

    # For each annotator, track agreement with LLM
    annotator_agreements = [0, 0, 0]  # For 3 annotators

    # Track distribution of LLM's most frequent answer vs human consensus
    # Categories passed in should already be sorted and include unknown
    all_categories = categories

    confusion_matrix = {
        expected: {predicted: 0 for predicted in all_categories}
        for expected in all_categories
    }

    from modality_llm.schema import Taxonomy

    tax = Taxonomy(taxonomy)
    for result in results:
        # --- get LLM prediction ---
        task = result.classification.get(tax)
        if not task or not task.english:
            raise ValueError(
                f"Missing classification result for taxonomy '{taxonomy}' in example '{result.eid}'"
            )
        llm_answer = max(task.english.distribution.items(), key=lambda x: x[1])[0]

        # --- get human consensus category ---
        if not result.expected_categories or tax not in result.expected_categories:
            raise ValueError(
                f"Missing expected_categories for taxonomy '{taxonomy}' in example '{result.eid}'"
            )
        exp_list = result.expected_categories[tax]
        if not exp_list:
            raise ValueError(
                f"No consensus categories for taxonomy '{taxonomy}' in example '{result.eid}'"
            )
        consensus = exp_list[0]

        # --- strict validation ---
        if consensus not in all_categories:
            raise ValueError(
                f"Unknown consensus category '{consensus}' for taxonomy '{taxonomy}' in example '{result.eid}'"
            )
        if llm_answer not in all_categories:
            raise ValueError(
                f"Unknown LLM answer category '{llm_answer}' for taxonomy '{taxonomy}' in example '{result.eid}'"
            )

        # --- update stats ---
        confusion_matrix[consensus][llm_answer] += 1
        if llm_answer == consensus:
            agreement_with_consensus += 1

        human_ann = result.human_annotations or []
        # agreement with any
        if isinstance(human_ann, list) and llm_answer in human_ann:
            agreement_with_any += 1
        # individual annotator agreement
        if isinstance(human_ann, list):
            for i, ann in enumerate(human_ann):
                if i >= 3:
                    break
                if llm_answer == ann:
                    annotator_agreements[i] += 1

    # If we couldn't process any results, exit early
    if total_examples == 0:
        print("No valid results to analyze.")
        return

    # Calculate agreement percentages
    consensus_agreement_pct = (agreement_with_consensus / total_examples) * 100
    any_agreement_pct = (agreement_with_any / total_examples) * 100
    annotator_agreement_pcts = [
        (count / total_examples) * 100 for count in annotator_agreements
    ]

    print(
        f"Consensus: {agreement_with_consensus}/{total_examples} ({consensus_agreement_pct:.1f}%)"
    )
    # Print results
    print(f"Total examples: {total_examples}")
    print(
        f"Agreement with consensus: {agreement_with_consensus} ({consensus_agreement_pct:.1f}%)"
    )
    print(
        f"Agreement with any annotator: {agreement_with_any} ({any_agreement_pct:.1f}%)"
    )
    print(
        f"Agreement with individual annotators: {annotator_agreement_pcts[0]:.1f}%, {annotator_agreement_pcts[1]:.1f}%, {annotator_agreement_pcts[2]:.1f}%"
    )

    # Print confusion matrix using original sorted categories
    print("\nConfusion Matrix (Expected vs LLM Prediction):")
    print("          ", end="")
    for category in categories:
        print(f"{category[:7]:>8}", end="")
    print()

    for expected in categories:
        print(f"{expected[:10]:<10}", end="")
        for predicted in categories:
            count = confusion_matrix.get(expected, {}).get(predicted, 0)
            print(f"{count:>8}", end="")
        print()

    # Calculate inter-annotator agreement (Fleiss' kappa)
    try:
        from modality_llm.schema import Taxonomy

        kappa_data: list[list[int]] = []
        tax = Taxonomy(taxonomy)
        for r in results:
            human_ann = (r.human_annotations or {}).get(tax, [])
            task = r.classification.get(tax)
            dist = task.english.distribution if task and task.english else {}
            llm_answer = max(dist.items(), key=lambda x: x[1])[0] if dist else None
            combined = list(human_ann) + ([llm_answer] if llm_answer else [])
            if not combined:
                continue
            counts = [combined.count(cat) for cat in categories]
            if sum(counts) > 0:
                kappa_data.append(counts)
        if kappa_data:
            import numpy as np

            kappa = fleiss_kappa(np.array(kappa_data))
            print(f"\nFleiss' kappa (including LLM as an annotator): {kappa:.3f}")

            # Interpret kappa
            if kappa < 0:
                interpretation = "Poor agreement"
            elif kappa < 0.2:
                interpretation = "Slight agreement"
            elif kappa < 0.4:
                interpretation = "Fair agreement"
            elif kappa < 0.6:
                interpretation = "Moderate agreement"
            elif kappa < 0.8:
                interpretation = "Substantial agreement"
            else:
                interpretation = "Almost perfect agreement"

            print(f"Interpretation: {interpretation}")
        else:
            print("\nNot enough valid data to calculate Fleiss' kappa.")
    except Exception as e:
        print(f"Could not calculate Fleiss' kappa: {e}")
        traceback.print_exc()

    # Build one combined dashboard: confusion‐matrix + bar‐chart
    try:
        sanitized_name = sanitize_model_name(model_name)
        viz = ConfusionMatrixUncertaintyViz(results, categories, taxonomy, model_name)
        confusion_dashboard = viz.create_dashboard()

        df_agree = pl.DataFrame(
            {
                "label": ["Consensus", "Any Annotator"]
                + [f"Annotator {i + 1}" for i in range(3)],
                "agreement": [consensus_agreement_pct, any_agreement_pct]
                + annotator_agreement_pcts,
            }
        )
        agreement_chart = (
            alt.Chart(df_agree)
            .mark_bar()
            .encode(
                x=alt.X("label:N", sort=None),
                y=alt.Y("agreement:Q", title="Agreement (%)"),
                tooltip=["label", "agreement"],
            )
            .properties(title=f"{sanitized_name} - LLM vs Human ({taxonomy.upper()})")
        )

        full_dashboard = alt.vconcat(
            confusion_dashboard, agreement_chart
        ).resolve_scale(color="independent")

        full_dashboard_name = f"{sanitized_name}_full_dashboard_{taxonomy}.html"
        full_dashboard.save(full_dashboard_name)
        print(f"\nFull dashboard saved to: {full_dashboard_name}")

        # --- WRITE OUT A CSV + MARKDOWN REPORT FOR EASY SHARING ---
        from modality_llm.analysis_reports import (
            write_modal_markdown_report,
            write_summary_csv,
        )

        # 1) Collect all of our printed metrics into a dict
        metrics = {
            "consensus_agreement_pct": round(consensus_agreement_pct, 2),
            "any_agreement_pct": round(any_agreement_pct, 2),
            "fleiss_kappa": round(kappa, 3) if "kappa" in locals() else None,
        }
        # add individual annotator agreements too
        for i, pct in enumerate(annotator_agreement_pcts[:3], start=1):
            metrics[f"annotator_{i}_agreement_pct"] = round(pct, 2)

        # 2) Reference our existing artifacts
        file_links = {
            "full_dashboard": full_dashboard_name,
            "uncertainty_report_csv": f"{sanitized_name}_modal_uncertainty_report.csv",
        }

        # 3) Dump a one-row CSV of metrics
        csv_name = f"{sanitized_name}_{taxonomy}_summary.csv"
        write_summary_csv(csv_name, metrics)
        print(f"Summary CSV saved to: {csv_name}")
        file_links["summary_csv"] = csv_name

        # 4) Finally, write the Markdown that ties it all together
        md_name = f"{sanitized_name}_{taxonomy}_report.md"
        write_modal_markdown_report(
            md_name,
            model_name,
            taxonomy,
            metrics,
            file_links,
        )
        print(f"Markdown report saved to: {md_name}")
    except Exception as e:
        print(f"Error creating combined dashboard: {e}")
        traceback.print_exc()

    # Call reliability analysis at the end
    analyze_classification_reliability(results, categories, taxonomy, model_name)


def analyze_classification_reliability(
    results: List[UnifiedResult],
    categories: List[str],
    taxonomy: Literal["palmer", "quirk"],
    model_name: str = "model",
) -> None:
    """
    Analyze classification reliability using sigma levels and uncertainty metrics.

    Args:
        results: List of UnifiedResult
        categories: List of categories
        taxonomy: 'palmer' or 'quirk'
        model_name: Name of the model
    """
    print(f"\n=== CLASSIFICATION RELIABILITY ANALYSIS ({taxonomy.upper()}) ===\n")

    # Calculate per-class reliability metrics
    class_metrics = {}

    tax_enum = Taxonomy(taxonomy)
    for category in categories:
        # collect all examples whose (possibly list‐valued) Expected == this category
        category_results: list[UnifiedResult] = []
        for r in results:
            exp_list = (r.expected_categories or {}).get(tax_enum, [])
            exp_val = exp_list[0] if exp_list else None
            if exp_val == category:
                category_results.append(r)

        if not category_results:
            continue

        # Calculate accuracy distribution
        accuracies = []
        for _ in range(100):  # Bootstrap
            idxs = np.random.choice(
                len(category_results), size=len(category_results), replace=True
            )
            sampled = [category_results[i] for i in idxs]
            correct = 0
            for r in sampled:
                task = r.classification.get(tax_enum)
                dist = task.english.distribution if task and task.english else {}
                pred = max(dist.items(), key=lambda x: x[1])[0] if dist else None
                if pred == category:
                    correct += 1
            accuracies.append(correct / len(sampled))

        class_metrics[category] = {
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "cv": np.std(accuracies) / (np.mean(accuracies) + 1e-10),
            "lower_95": np.percentile(accuracies, 2.5),
            "upper_95": np.percentile(accuracies, 97.5),
        }

    # Print reliability report
    print("Per-Class Reliability Metrics:")
    print("-" * 60)
    for category, metrics in class_metrics.items():
        print(f"\n{category}:")
        print(
            f"  Mean Accuracy: {metrics['mean_accuracy']:.3f} ± {metrics['std_accuracy']:.3f}"
        )
        print(f"  95% CI: [{metrics['lower_95']:.3f}, {metrics['upper_95']:.3f}]")
        print(f"  Coefficient of Variation: {metrics['cv']:.3f}")

        # Sigma level assessment
        if metrics["cv"] < 0.1:
            reliability = "Excellent (>3σ)"
        elif metrics["cv"] < 0.2:
            reliability = "Good (2-3σ)"
        elif metrics["cv"] < 0.3:
            reliability = "Fair (1-2σ)"
        else:
            reliability = "Poor (<1σ)"
        print(f"  Reliability Level: {reliability}")
