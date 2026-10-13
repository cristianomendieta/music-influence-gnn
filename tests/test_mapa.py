"""Unit tests for evaluation/mapa.py (Phase 6b, MAP-04..MAP-07)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from music_diffusion_gnn.evaluation.mapa import (
    build_map,
    compare_cell,
    holm_family,
    label_cell,
    structure_pays,
)
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


# -------------------------------------- MAP-04..06: the map table and "estrutura paga"

KS = (1, 2, 4)
SPLITS = ("current", "pre_pandemia")
N_SONGS = 40


def _amp(rng, *, control_worse: bool) -> float:
    """Control error amplitude: clearly worse than the proposal (0.01), or equal to it."""
    return (0.02 if control_worse else 0.01) + rng.normal(0, 0.0004)


def _world(plan: dict[tuple[str, str], dict[int, tuple[bool, bool]]], n_songs=None, seed=0):
    """Synthetic predictions and strata.

    ``plan[(split, estrato)][k] = (wins_sem_grafo, wins_embaralhado)``. ``y_true`` is 0, so
    a model's squared error is its ``y_pred`` squared.
    """
    rng = np.random.default_rng(seed)
    n_songs = n_songs or {}
    rows, strata = [], []
    for (split, estrato), by_k in plan.items():
        n = n_songs.get((split, estrato), N_SONGS)
        for i in range(n):
            song = f"{estrato}-{i}"
            strata.append({"song_id": song, "chart": "top200", "origin_week": 100, "estrato": estrato})
            for k in KS:
                win_sem, win_emb = by_k[k]
                amps = {"proposta": 0.01 + rng.normal(0, 0.0004),
                        "sem_grafo": _amp(rng, control_worse=win_sem),
                        "embaralhado": _amp(rng, control_worse=win_emb),
                        "persistencia": 0.5}
                for modelo, amp in amps.items():
                    rows.append({"song_id": song, "chart": "top200", "origin_week": 100, "k": k,
                                 "split_regime": split, "modelo": modelo, "y_true": 0.0,
                                 "y_pred": amp, "onchart": True})
    return pd.DataFrame(rows), pd.DataFrame(strata).drop_duplicates()


WIN = (True, True)
TIE = (False, False)


def _plan_all_splits(estrato, by_k):
    return {(s, estrato): by_k for s in SPLITS}


def test_structure_pays_marks_the_stratum_that_wins_both_controls_in_two_horizons_in_both_splits():
    plan = {**_plan_all_splits("estreia", {1: WIN, 2: WIN, 4: TIE}),
            **_plan_all_splits("maduro_mono", {1: TIE, 2: TIE, 4: TIE})}
    preds, strata = _world(plan)
    table = build_map(preds, strata, B=B, seed=1)
    verdict = structure_pays(table).set_index("estrato")["veredito"]
    assert verdict["estreia"] == "estrutura paga"
    assert verdict["maduro_mono"] == "nao paga"


def test_winning_in_only_one_split_is_not_estrutura_paga():
    plan = {("current", "jovem"): {1: WIN, 2: WIN, 4: WIN},
            ("pre_pandemia", "jovem"): {1: TIE, 2: TIE, 4: TIE}}
    preds, strata = _world(plan)
    verdict = structure_pays(build_map(preds, strata, B=B, seed=1)).set_index("estrato")["veredito"]
    assert verdict["jovem"] == "nao paga"


def test_winning_only_one_control_is_not_estrutura_paga():
    only_sem = (True, False)
    plan = _plan_all_splits("reentrada", {1: only_sem, 2: only_sem, 4: only_sem})
    preds, strata = _world(plan)
    verdict = structure_pays(build_map(preds, strata, B=B, seed=1)).set_index("estrato")["veredito"]
    assert verdict["reentrada"] == "nao paga"


def test_a_stratum_too_small_in_one_split_is_nao_testavel_never_estrutura_paga():
    plan = _plan_all_splits("estreia", {1: WIN, 2: WIN, 4: WIN})
    preds, strata = _world(plan, n_songs={("pre_pandemia", "estreia"): 10})
    table = build_map(preds, strata, B=B, seed=1)
    assert structure_pays(table).set_index("estrato")["veredito"]["estreia"] == "nao testavel"
    small = table[(table.split_regime == "pre_pandemia") & (table.estrato == "estreia")]
    assert (small.rotulo == "insuficiente").all() and len(small) == len(KS) * 2


def test_table_has_one_row_per_split_stratum_horizon_and_control():
    plan = {**_plan_all_splits("estreia", {1: WIN, 2: WIN, 4: WIN}),
            **_plan_all_splits("jovem", {1: TIE, 2: TIE, 4: TIE})}
    preds, strata = _world(plan)
    table = build_map(preds, strata, B=B, seed=1)
    assert len(table) == 2 * 2 * 3 * 2
    assert not table.duplicated(["split_regime", "estrato", "k", "contraste"]).any()
    assert set(table.contraste) == {"sem_grafo", "embaralhado"}


def test_every_label_is_recomputable_from_its_own_row():
    plan = {**_plan_all_splits("estreia", {1: WIN, 2: TIE, 4: WIN}),
            **_plan_all_splits("jovem", {1: TIE, 2: TIE, 4: TIE})}
    preds, strata = _world(plan)
    table = build_map(preds, strata, B=B, seed=1)
    tested = table[~table.insuficiente]
    assert len(tested) == len(table)
    for r in tested.itertuples():
        assert r.rotulo == label_cell(r.ci_lo, r.ci_hi, r.p_holm, r.margin)
    # and the designed wins are labelled as wins
    cell = table[(table.split_regime == "current") & (table.estrato == "estreia") & (table.k == 1)]
    assert set(cell.rotulo) == {"vence"}


def test_holm_in_the_table_is_one_family_per_split():
    plan = {**_plan_all_splits("estreia", {1: WIN, 2: TIE, 4: TIE}),
            **_plan_all_splits("jovem", {1: TIE, 2: TIE, 4: TIE})}
    preds, strata = _world(plan)
    table = build_map(preds, strata, B=B, seed=1)
    for _, grp in table.groupby("split_regime"):
        assert grp.p_holm.to_numpy() == pytest.approx(holm_correction(grp.p_value.to_numpy()))


def test_only_on_chart_rows_and_only_the_three_neural_models_are_used():
    plan = _plan_all_splits("estreia", {1: WIN, 2: WIN, 4: WIN})
    preds, strata = _world(plan)
    base = build_map(preds, strata, B=B, seed=1)
    off = preds.copy()
    off["onchart"] = False
    off["y_pred"] = 9.0                          # absurd errors, but off-chart
    noisy = pd.concat([preds, off], ignore_index=True)
    again = build_map(noisy, strata, B=B, seed=1)
    assert again[["mean_diff", "ci_lo", "ci_hi", "p_value"]].to_numpy() == pytest.approx(
        base[["mean_diff", "ci_lo", "ci_hi", "p_value"]].to_numpy())


def test_origins_without_a_stratum_or_sem_serie_are_left_out():
    plan = _plan_all_splits("estreia", {1: WIN, 2: WIN, 4: WIN})
    preds, strata = _world(plan)
    strata.loc[strata.index[:5], "estrato"] = "sem_serie"
    table = build_map(preds, strata, B=B, seed=1)
    assert set(table.estrato) == {"estreia"}
    assert table.n_musicas.max() == N_SONGS - 5


def test_a_song_in_both_charts_counts_once_in_n_musicas():
    plan = _plan_all_splits("estreia", {1: WIN, 2: WIN, 4: WIN})
    preds, strata = _world(plan)
    other = preds.assign(chart="viral50")
    other_strata = strata.assign(chart="viral50")
    both = pd.concat([preds, other], ignore_index=True)
    both_strata = pd.concat([strata, other_strata], ignore_index=True)
    table = build_map(both, both_strata, B=B, seed=1)
    assert (table.n_musicas == N_SONGS).all()
    assert (table.n_origens == 2 * N_SONGS).all()
