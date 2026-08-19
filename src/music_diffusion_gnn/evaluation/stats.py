"""Evaluation stats: paired Wilcoxon, bootstrap CI, directional accuracy."""
from __future__ import annotations

import numpy as np
import pandas as pd
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


def paired_by_song(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    song_ids: np.ndarray,
    *,
    B: int = 10000,
    seed: int,
) -> dict[str, float]:
    """Paired Wilcoxon + bootstrap CI on per-song mean error, not per-origin.

    Multiple origins from the same song are not independent observations —
    pairing the test directly on origins inflates the sample size (and
    significance) far beyond the true number of songs. Origin-level errors
    are averaged per song first; the test and CI then run on that per-song
    series, matching Mode 1's genuinely per-song comparison.

    Returns the ``wilcoxon_signed_rank`` dict plus ``ci_lo``/``ci_hi``/
    ``mean_diff`` (bootstrap CI of the per-song mean difference), ``win_rate``
    (fraction of songs where ``a < b``), and ``n_songs``.
    """
    df = pd.DataFrame({
        "song_id": np.asarray(song_ids),
        "a": np.asarray(errors_a, dtype=float),
        "b": np.asarray(errors_b, dtype=float),
    })
    per_song = df.groupby("song_id", observed=True)[["a", "b"]].mean()
    a = per_song["a"].to_numpy()
    b = per_song["b"].to_numpy()
    w = wilcoxon_signed_rank(a, b)
    lo, hi, md = bootstrap_ci_diff(a, b, B=B, seed=seed)
    return {
        **w,
        "ci_lo": lo,
        "ci_hi": hi,
        "mean_diff": md,
        "win_rate": float((a < b).mean()),
        "n_songs": len(per_song),
    }


def holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction for a family of p-values.

    Returns adjusted p-values in the same order/shape as the input, each
    clipped to <= 1.0. Controls the family-wise error rate across a set of
    comparisons run together (e.g. all chart × horizon cells in one report).

    A NaN input (a failed comparison — e.g. too few paired songs) stays NaN
    in the output and is excluded from the ranking of the other p-values;
    Python's ``max(x, nan)`` silently keeps ``x`` rather than propagating the
    NaN, so this can't just fall out of the naive step-down loop.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.full(m, np.nan)
    valid = ~np.isnan(p)
    p_valid = p[valid]
    m_valid = len(p_valid)
    order = np.argsort(p_valid)
    valid_idx = np.flatnonzero(valid)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m_valid - rank) * p_valid[idx])
        adjusted[valid_idx[idx]] = min(running_max, 1.0)
    return adjusted


def aggregate_seeds(values: np.ndarray) -> tuple[float, float]:
    """Mean and sample std (ddof=1) of a metric across seeds.

    Returns ``(mean, 0.0)`` for a single value — there is no spread to
    report from one seed.
    """
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean, std


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
