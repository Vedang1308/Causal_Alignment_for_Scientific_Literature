"""
Statistical Tests, Bootstrapped Confidence Intervals & Calibration Metrics.
Implements Section V-B:
- 1,000-sample bootstrap 95% confidence intervals
- Expected Calibration Error (ECE)
- Paired bootstrap hypothesis significance testing (p-values)
"""

from typing import Callable, List, Tuple, Dict, Any
import numpy as np


def bootstrap_confidence_interval(
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes point estimate and bootstrapped confidence interval [ci_lower, ci_upper].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    point_estimate = float(metric_func(y_true, y_pred))
    rng = np.random.default_rng(random_state)

    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, size=n)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        score = metric_func(sample_true, sample_pred)
        bootstrapped_scores.append(score)

    bootstrapped_scores = np.sort(bootstrapped_scores)
    alpha = (1.0 - ci) / 2.0
    lower_idx = int(alpha * n_bootstraps)
    upper_idx = int((1.0 - alpha) * n_bootstraps)

    lower = float(bootstrapped_scores[max(0, lower_idx)])
    upper = float(bootstrapped_scores[min(n_bootstraps - 1, upper_idx)])

    return point_estimate, lower, upper


def paired_bootstrap_p_value(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> float:
    """
    Paired bootstrap hypothesis test for difference in means (H0: mean(A) == mean(B)).
    Returns two-sided p-value.
    """
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    observed_diff = np.mean(diffs)
    if observed_diff == 0:
        return 1.0

    # Center under null hypothesis
    centered_diffs = diffs - observed_diff
    n = len(diffs)
    rng = np.random.default_rng(random_state)

    count = 0
    for _ in range(n_bootstraps):
        boot_sample = rng.choice(centered_diffs, size=n, replace=True)
        if abs(np.mean(boot_sample)) >= abs(observed_diff):
            count += 1

    return float((count + 1) / (n_bootstraps + 1))


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) across confidence bins.
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    confidences = np.asarray(confidences)
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)

    N = len(y_true)
    if N == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(predictions[in_bin] == y_true[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / N) * abs(bin_acc - bin_conf)

    return float(ece)
