"""Unit tests for evaluation/stats.py (T3)."""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from music_diffusion_gnn.evaluation.stats import (
    bootstrap_ci_diff,
    bootstrap_ci_mean,
    directional_accuracy,
    wilcoxon_signed_rank,
)


def test_wilcoxon_signed_rank_matches_scipy_direct_call():
    rmse_a = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    rmse_b = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    result = wilcoxon_signed_rank(rmse_a, rmse_b)
    expected = scipy_stats.wilcoxon(rmse_a, rmse_b, zero_method="wilcox")
    assert result["statistic"] == float(expected.statistic)
    assert result["p_value"] == float(expected.pvalue)
    assert result["n"] == len(rmse_a)


def test_wilcoxon_signed_rank_one_sided_difference_has_small_p_value():
    rng = np.random.default_rng(0)
    rmse_a = 10.0 + rng.normal(0, 0.1, size=20)
    rmse_b = 1.0 + rng.normal(0, 0.1, size=20)
    result = wilcoxon_signed_rank(rmse_a, rmse_b)
    assert result["p_value"] < 0.01
    assert result["n"] == 20


def test_wilcoxon_signed_rank_asserts_equal_length():
    rmse_a = np.array([1.0, 2.0, 3.0])
    rmse_b = np.array([1.0, 2.0])
    try:
        wilcoxon_signed_rank(rmse_a, rmse_b)
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_bootstrap_ci_mean_is_deterministic_for_fixed_seed():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result1 = bootstrap_ci_mean(values, B=500, seed=42)
    result2 = bootstrap_ci_mean(values, B=500, seed=42)
    assert result1 == result2


def test_bootstrap_ci_mean_brackets_sample_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    lo, hi, mean = bootstrap_ci_mean(values, B=2000, seed=1)
    assert mean == values.mean()
    assert lo <= mean <= hi


def test_bootstrap_ci_diff_is_deterministic_for_fixed_seed():
    rmse_a = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    rmse_b = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    result1 = bootstrap_ci_diff(rmse_a, rmse_b, B=500, seed=7)
    result2 = bootstrap_ci_diff(rmse_a, rmse_b, B=500, seed=7)
    assert result1 == result2


def test_bootstrap_ci_diff_brackets_mean_diff():
    rmse_a = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    rmse_b = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    lo, hi, mean_diff = bootstrap_ci_diff(rmse_a, rmse_b, B=2000, seed=3)
    assert mean_diff == (rmse_a - rmse_b).mean()
    assert lo <= mean_diff <= hi


def test_directional_accuracy_known_fraction():
    y_origin = np.array([10.0, 10.0, 10.0, 10.0])
    y_true_k = np.array([12.0, 8.0, 10.0, 15.0])
    y_pred_k = np.array([11.0, 9.0, 10.0, 9.0])
    # case 0: true up, pred up -> match
    # case 1: true down, pred down -> match
    # case 2: true flat, pred flat -> match
    # case 3: true up, pred down -> mismatch
    assert directional_accuracy(y_origin, y_true_k, y_pred_k) == 0.75


def test_directional_accuracy_handles_zero_delta_without_crash():
    y_origin = np.array([5.0, 5.0])
    y_true_k = np.array([5.0, 5.0])
    y_pred_k = np.array([5.0, 6.0])
    assert directional_accuracy(y_origin, y_true_k, y_pred_k) == 0.5
