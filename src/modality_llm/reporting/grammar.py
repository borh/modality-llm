import json
from typing import List

from modality_llm.analysis_grammar_bayes import (
    bayesian_analysis,
    bayesian_analysis_with_pymc,
)


def analyze_grammar_results(
    results_json_path: str,
    taxonomy: str,
    model_name: str,
) -> None:
    """
    Load the saved JSON results and re-run the grammar Bayesian analyses.
    """
    from modality_llm.schema import UnifiedResult

    with open(results_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    results: List[UnifiedResult] = [UnifiedResult.model_validate(d) for d in raw]

    # Run all the main analyses
    bayesian_analysis(results)
    bayesian_analysis_with_pymc(
        results,
        model_name=model_name,
        use_advi=True,
        n_samples=500,
    )
    # Optionally, could call hierarchical analysis if EID/Tested_Modal present
    # bayesian_analysis_hierarchical_grammar(results, language="English")
    # bayesian_analysis_hierarchical_grammar(results, language="Japanese")
