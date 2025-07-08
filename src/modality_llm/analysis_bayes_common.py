"""
Common Bayesian analysis utilities and models.
"""

from typing import Any, Dict, Optional, Tuple

import pymc as pm
from scipy.stats import beta


def interpret_bayes_factor(bf: float) -> str:
    """
    Interpret the strength of evidence from a Bayes factor.

    Examples:
        >>> interpret_bayes_factor(50)
        'Very strong evidence'
        >>> interpret_bayes_factor(0.5)
        'No evidence'
    """
    if bf > 100:
        return "Extreme evidence"
    elif bf > 30:
        return "Very strong evidence"
    elif bf > 10:
        return "Strong evidence"
    elif bf > 3:
        return "Moderate evidence"
    elif bf > 1:
        return "Anecdotal evidence"
    else:
        return "No evidence"


def calculate_bayes_factor(
    distribution: Dict[str, int], expected: str, epsilon: float = 1e-9
) -> float:
    """
    Calculate Bayes factor for a given distribution and expected answer.

    Examples:
        >>> calculate_bayes_factor({'yes': 2, 'no': 1}, 'yes')
        2.0
        >>> calculate_bayes_factor({'yes': 2, 'no': 1}, 'no')
        0.5
    """
    # avoid adding epsilon when count > 0
    yes_count = distribution.get("yes", 0)
    no_count = distribution.get("no", 0)
    if yes_count == 0 and no_count == 0:
        return epsilon
    if expected == "yes":
        denom = no_count or epsilon
        return yes_count / denom
    else:
        denom = yes_count or epsilon
        return no_count / denom


def beta_posterior_analysis(
    distribution: Dict[str, int], prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate posterior mean and 95% credible interval using Beta distribution.

    Examples:
        >>> mean, (low, high) = beta_posterior_analysis({'yes': 2, 'no': 3}, prior_alpha=1, prior_beta=1)
        >>> round(mean, 3)
        0.429

    Returns:
        (posterior_mean, (ci_lower, ci_upper))
    """
    yes_count = distribution.get("yes", 0)
    no_count = distribution.get("no", 0)
    alpha_post = prior_alpha + yes_count
    beta_post = prior_beta + no_count

    posterior_mean = alpha_post / (alpha_post + beta_post)
    credible_interval = beta.interval(0.95, alpha_post, beta_post)

    return posterior_mean, credible_interval


def run_inference(
    model: pm.Model,
    use_advi: bool = True,
    n_samples: int = 500,
    chains: int = 8,
    cores: Optional[int] = None,
) -> Any:
    """Run inference on a PyMC model using either ADVI or MCMC."""
    with model:
        if use_advi:
            print(f"Using ADVI with {n_samples * 4} iterations...")
            approx = pm.fit(n=n_samples * 4, method="advi", random_seed=42)
            trace = approx.sample(n_samples, random_seed=42)
            print("ADVI completed")
        else:
            print(f"Using MCMC with {n_samples} samples, {chains} chains...")
            trace = pm.sample(
                n_samples,
                tune=n_samples,
                chains=chains,
                cores=cores,
                return_inferencedata=True,
                random_seed=42,
            )
    return trace
