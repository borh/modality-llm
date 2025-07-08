import json
from typing import List

from modality_llm.analysis_compare import compare_with_human_annotators
from modality_llm.analysis_modal_bayes import bayesian_analysis_modal
from modality_llm.modal import get_categories


def analyze_modal_results(
    results_json_path: str,
    taxonomy: str,
    model_name: str,
) -> None:
    """
    Load the saved JSON results and re-run the human‐comparison
    and Bayesian‐analysis reporting (print+plots).
    """
    from modality_llm.schema import UnifiedResult

    with open(results_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    results: List[UnifiedResult] = [UnifiedResult.model_validate(d) for d in raw]

    categories = get_categories(taxonomy)

    # original side‐effecting calls:
    compare_with_human_annotators(results, categories, taxonomy, model_name)
    bayesian_analysis_modal(
        results=results,
        categories=categories,
        model_name=model_name,
        taxonomy=taxonomy,
    )
