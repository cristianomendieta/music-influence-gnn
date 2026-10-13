"""Unit tests for evaluation/mapa.py (Phase 6b, MAP-04..MAP-07)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from music_diffusion_gnn.evaluation.mapa import compare_cell, holm_family, label_cell
from music_diffusion_gnn.evaluation.stats import holm_correction, paired_by_song

B = 500


def _synthetic(n_songs=40, per_song=3, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    songs = np.repeat([f"s{i}" for i in range(n_songs)], per_song)
    b = rng.uniform(0.01, 0.05, size=len(songs))
    a = b + shift + rng.normal(0, 0.002, size=len(songs))
    return a, b, songs


# ------------------------------------------------------------ MAP-04: one cell

def test_hand_computed_cell_values():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.2, 0.4, 0.6])
    r = compare_cell(a, b, np.array(["s1", "s2", "s3"]), B=B, seed=1)
    assert r["mean_diff"] == pytest.approx(-0.2)
    assert r["win_rate"] == 1.0
    assert r["n_songs"] == 3
    assert r["mean_b"] == pytest.approx(0.4)
    assert r["margin"] == pytest.approx(0.05 * 0.4)


def test_ci_and_p_match_paired_by_song_with_same_seed():
    a, b, songs = _synthetic(shift=-0.003)
    r = compare_cell(a, b, songs, B=B, seed=7)
    ref = paired_by_song(a, b, songs, B=B, seed=7)
    assert r["mean_diff"] == pytest.approx(ref["mean_diff"])
    assert r["ci_lo"] == pytest.approx(ref["ci_lo"])
    assert r["ci_hi"] == pytest.approx(ref["ci_hi"])
    assert r["p_value"] == pytest.approx(ref["p_value"])
    assert r["win_rate"] == pytest.approx(ref["win_rate"])


def test_margin_is_equiv_frac_times_mean_per_song_error_of_the_control():
    a, b, songs = _synthetic()
    r = compare_cell(a, b, songs, B=B, seed=1, equiv_frac=0.10)
    per_song_b = np.array([b[songs == s].mean() for s in np.unique(songs)])
    assert r["margin"] == pytest.approx(0.10 * per_song_b.mean())


def test_a_song_with_origins_in_both_charts_is_one_observation():
    # s1 has two origins (e.g. viral50 and top200): its per-song means are a=0.2, b=0.4
    a = np.array([0.1, 0.3, 0.1, 0.5])
    b = np.array([0.3, 0.5, 0.2, 0.6])
    songs = np.array(["s1", "s1", "s2", "s3"])
    r = compare_cell(a, b, songs, B=B, seed=1)
    assert r["n_songs"] == 3
    assert r["mean_diff"] == pytest.approx(np.mean([0.2 - 0.4, 0.1 - 0.2, 0.5 - 0.6]))


def test_identical_errors_give_p_one_and_zero_difference_without_raising():
    _a, b, songs = _synthetic()
    r = compare_cell(b, b, songs, B=B, seed=1)
    assert r["p_value"] == 1.0
    assert r["mean_diff"] == 0.0
    assert (r["ci_lo"], r["ci_hi"]) == (0.0, 0.0)


def test_fixed_seed_makes_the_ci_reproducible():
    a, b, songs = _synthetic(shift=-0.002)
    r1 = compare_cell(a, b, songs, B=B, seed=11)
    r2 = compare_cell(a, b, songs, B=B, seed=11)
    assert (r1["ci_lo"], r1["ci_hi"]) == (r2["ci_lo"], r2["ci_hi"])


# ------------------------------------------------------ MAP-05: labels and Holm family

def test_vence_needs_ci_below_zero_and_holm_below_alpha():
    assert label_cell(-0.010, -0.002, 0.01, margin=0.001) == "vence"


def test_perde_is_the_mirror_of_vence():
    assert label_cell(0.002, 0.010, 0.01, margin=0.001) == "perde"


def test_ci_below_zero_with_holm_not_significant_is_not_vence():
    assert label_cell(-0.010, -0.002, 0.30, margin=0.001) == "inconclusivo"


def test_holm_exactly_at_alpha_is_not_significant():
    assert label_cell(-0.010, -0.002, 0.05, margin=0.001) == "inconclusivo"


def test_ci_touching_zero_is_not_vence():
    assert label_cell(-0.010, 0.0, 0.001, margin=0.001) == "inconclusivo"


def test_ci_inside_the_margin_without_a_significant_effect_is_equivalente():
    assert label_cell(-0.001, 0.001, 0.90, margin=0.002) == "equivalente"


def test_ci_exactly_on_the_margin_still_counts_as_equivalente():
    assert label_cell(-0.002, 0.002, 0.90, margin=0.002) == "equivalente"


def test_wide_ci_outside_the_margin_is_inconclusivo():
    assert label_cell(-0.010, 0.010, 0.90, margin=0.002) == "inconclusivo"


def test_a_significant_win_inside_the_margin_is_vence_not_equivalente():
    assert label_cell(-0.0015, -0.0005, 0.001, margin=0.002) == "vence"


def _tests_frame():
    return pd.DataFrame({
        "split_regime": ["current"] * 4 + ["pre_pandemia"] * 3,
        "p_value": [0.001, 0.02, 0.04, 0.30, 0.01, 0.20, 0.50],
        "insuficiente": [False, False, False, False, False, False, False],
    })


def test_holm_family_is_one_correction_per_split():
    df = _tests_frame()
    got = holm_family(df)
    assert got[:4].to_numpy() == pytest.approx(holm_correction(df.p_value[:4].to_numpy()))
    assert got[4:].to_numpy() == pytest.approx(holm_correction(df.p_value[4:].to_numpy()))


def test_insuficiente_cells_are_nan_and_leave_the_family_of_the_others():
    df = _tests_frame()
    df.loc[3, "insuficiente"] = True
    got = holm_family(df)
    assert np.isnan(got[3])
    assert got[:3].to_numpy() == pytest.approx(holm_correction(df.p_value[:3].to_numpy()))


def test_a_full_family_of_thirty_cells_is_corrected_together():
    p = np.linspace(0.001, 0.9, 30)
    df = pd.DataFrame({"split_regime": "current", "p_value": p, "insuficiente": False})
    assert holm_family(df).to_numpy() == pytest.approx(holm_correction(p))
    assert holm_family(df).iloc[0] == pytest.approx(min(p[0] * 30, 1.0))
