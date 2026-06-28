"""Evaluation stats: paired Wilcoxon, bootstrap CI, directional accuracy."""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_signed_rank(rmse_a: np.ndarray, rmse_b: np.ndarray) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test between two RMSE distributions.

    Returns dict with keys: statistic, p_value, n (number of pairs).
    """
    rmse_a = np.asarray(rmse_a, dtype=float)
    rmse_b = np.asarray(rmse_b, dtype=float)
    assert len(rmse_a) == len(rmse_b)
    result = stats.wilcoxon(rmse_a, rmse_b, zero_method="wilcox")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": len(rmse_a)}


def bootstrap_ci_mean(
    values: np.ndarray, *, B: int = 10000, seed: int
) -> tuple[float, float, float]:
    """Percentile bootstrap CI95% for the mean of `values`.

    Returns (lo, hi, mean) where mean is the original sample's mean.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float(values.mean())


def bootstrap_ci_diff(
    rmse_a: np.ndarray, rmse_b: np.ndarray, *, B: int = 10000, seed: int
) -> tuple[float, float, float]:
    """Paired percentile bootstrap CI95% for mean(rmse_a - rmse_b).

    Each resample draws one shared set of indices applied to both arrays,
    preserving the by-song pairing. Returns (lo, hi, mean_diff) where
    mean_diff is the original (unresampled) mean difference.
    """
    rmse_a = np.asarray(rmse_a, dtype=float)
    rmse_b = np.asarray(rmse_b, dtype=float)
    assert len(rmse_a) == len(rmse_b)
    rng = np.random.default_rng(seed)
    n = len(rmse_a)
    diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (rmse_a[idx] - rmse_b[idx]).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float((rmse_a - rmse_b).mean())


def directional_accuracy(
    y_origin: np.ndarray, y_true_k: np.ndarray, y_pred_k: np.ndarray
) -> float:
    """Fraction of cases where predicted and actual direction of change agree.

    Direction is sign(y_k - y_origin). A zero delta (no change) matches only
    another zero delta, which falls out naturally from elementwise sign comparison.
    """
    y_origin = np.asarray(y_origin, dtype=float)
    y_true_k = np.asarray(y_true_k, dtype=float)
    y_pred_k = np.asarray(y_pred_k, dtype=float)
    true_dir = np.sign(y_true_k - y_origin)
    pred_dir = np.sign(y_pred_k - y_origin)
    return float(np.mean(true_dir == pred_dir))
