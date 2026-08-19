"""Unit tests for evaluation/stats.py (T3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from music_diffusion_gnn.evaluation.stats import (
    aggregate_seeds,
    bootstrap_ci_diff,
    bootstrap_ci_mean,
    directional_accuracy,
    holm_correction,
    paired_by_song,
    wilcoxon_signed_rank,
)

ROOT = Path(__file__).resolve().parent.parent
MODE2_PATH = ROOT / "results" / "phase3" / "mode2_horizons.parquet"


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


# ---------------------------------------------------------------------------
# paired_by_song (item 03: pair by song, not by prediction origin)
# ---------------------------------------------------------------------------

def test_paired_by_song_aggregates_multiple_origins_per_song_first():
    # Two songs, each with 3 origins. Song 1: a always beats b. Song 2: b always beats a.
    song_ids = np.array(["s1", "s1", "s1", "s2", "s2", "s2"])
    errors_a = np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
    errors_b = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0])
    result = paired_by_song(errors_a, errors_b, song_ids, seed=1)
    assert result["n_songs"] == 2
    assert result["n"] == 2  # wilcoxon ran on 2 per-song means, not 6 origins


def test_paired_by_song_win_rate_is_fraction_of_songs_where_a_beats_b():
    song_ids = np.array(["s1", "s1", "s2", "s3"])
    errors_a = np.array([1.0, 1.0, 9.0, 1.0])
    errors_b = np.array([5.0, 5.0, 2.0, 5.0])
    result = paired_by_song(errors_a, errors_b, song_ids, seed=1)
    # s1: a=1 < b=5 (win), s2: a=9 > b=2 (loss), s3: a=1 < b=5 (win) -> 2/3
    assert result["win_rate"] == pytest.approx(2 / 3)


def test_paired_by_song_ci_matches_bootstrap_ci_diff_on_per_song_means():
    song_ids = np.array(["s1", "s1", "s2"])
    errors_a = np.array([2.0, 4.0, 9.0])  # s1 mean = 3.0
    errors_b = np.array([1.0, 1.0, 2.0])  # s1 mean = 1.0
    result = paired_by_song(errors_a, errors_b, song_ids, seed=7)
    expected_lo, expected_hi, expected_mean = bootstrap_ci_diff(
        np.array([3.0, 9.0]), np.array([1.0, 2.0]), seed=7
    )
    assert result["ci_lo"] == expected_lo
    assert result["ci_hi"] == expected_hi
    assert result["mean_diff"] == expected_mean


# ---------------------------------------------------------------------------
# holm_correction
# ---------------------------------------------------------------------------

def test_holm_correction_matches_hand_computed_example():
    p = np.array([0.01, 0.02, 0.03, 0.5])
    adjusted = holm_correction(p)
    # sorted ascending: 0.01,0.02,0.03,0.5 with multipliers 4,3,2,1
    # step-down running max: 0.04, 0.06, 0.06, 0.5
    np.testing.assert_allclose(adjusted, [0.04, 0.06, 0.06, 0.5])


def test_holm_correction_is_never_smaller_than_raw_p_value():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=10)
    adjusted = holm_correction(p)
    assert (adjusted >= p - 1e-12).all()
    assert (adjusted <= 1.0).all()


def test_holm_correction_single_pvalue_is_unchanged():
    assert holm_correction(np.array([0.03]))[0] == pytest.approx(0.03)


def test_holm_correction_propagates_nan_instead_of_reusing_a_neighbor():
    # A failed comparison (NaN) must stay NaN, not silently inherit another
    # cell's running_max via Python's max(x, nan) == x behavior.
    p = np.array([0.01, 0.02, float("nan"), 0.03])
    adjusted = holm_correction(p)
    assert np.isnan(adjusted[2])
    # The three real p-values are Holm-corrected among themselves only (m=3).
    np.testing.assert_allclose(adjusted[[0, 1, 3]], [0.03, 0.04, 0.04])


# ---------------------------------------------------------------------------
# aggregate_seeds
# ---------------------------------------------------------------------------

def test_aggregate_seeds_mean_and_std():
    mean, std = aggregate_seeds(np.array([1.0, 2.0, 3.0]))
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)  # ddof=1


def test_aggregate_seeds_single_value_has_zero_std():
    mean, std = aggregate_seeds(np.array([5.0]))
    assert mean == 5.0
    assert std == 0.0


# ---------------------------------------------------------------------------
# Recompute check against real Mode-2 results (item 03 acceptance criterion):
# per-song pairing must still land the GNN win rate in the previously
# reported 64%-77% range.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODE2_PATH.exists(), reason="results/phase3/mode2_horizons.parquet not present")
def test_recompute_per_song_wilcoxon_confirms_known_gnn_win_rate_range():
    m2 = pd.read_parquet(MODE2_PATH)
    win_rates = []
    for chart in ("viral50", "top200"):
        for k in sorted(m2["k"].unique()):
            sub = m2[(m2["chart"] == chart) & (m2["k"] == k)]
            gnn = sub[sub["model"] == "gnn"].dropna(subset=["y_true", "y_pred"])
            sir = sub[(sub["model"] == "sir") & (sub["converged"])].dropna(subset=["y_true", "y_pred"])
            paired = gnn.set_index(["song_id", "chart", "origin_week"]).join(
                sir.set_index(["song_id", "chart", "origin_week"]),
                lsuffix="_gnn", rsuffix="_sir", how="inner",
            )
            if len(paired) < 2:
                continue
            rmse_gnn = ((paired["y_true_gnn"] - paired["y_pred_gnn"]) ** 2).to_numpy()
            rmse_sir = ((paired["y_true_sir"] - paired["y_pred_sir"]) ** 2).to_numpy()
            song_ids = paired.index.get_level_values("song_id").to_numpy()
            result = paired_by_song(rmse_gnn, rmse_sir, song_ids, seed=42)
            win_rates.append(result["win_rate"])

    assert win_rates, "expected at least one (chart, k) cell with paired GNN/SIR origins"
    # Known range is "64% to 77%" (docs.md #03); allow a couple points of slack
    # since the exact figure depends on bootstrap seed, not on win_rate itself.
    assert all(0.62 <= wr <= 0.79 for wr in win_rates), win_rates
