"""Unit tests for evaluation/veredito_atencao.py (Phase 7, ATT-05).

Rules under test come from ``docs/adr/0006-precompromisso-fase-atencao.md``.
Synthetic predictions only (pandas/numpy, no torch).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from music_diffusion_gnn.evaluation.stats import holm_correction, paired_by_song
from music_diffusion_gnn.evaluation.veredito_atencao import (
    CONTRASTS,
    MAJORITY,
    N_CONFIGS_ATTENTION,
    N_CONFIGS_SAGE,
    SAGE_N_PARAMS,
    TIE_VAL_MSE,
    TREATED,
    budget_block,
    classify_outcome,
    compare_models,
    sensitivity_by_seed,
    verdict,
)

B = 200
CHARTS = ("viral50", "top200")
KS = (1, 2, 4)
SPLITS = ("current", "pre_pandemia")
MODELS = ("atencao", "sem_grafo", "atencao_emb", "proposta")


def _preds(noise=None, n_songs=40, origins=2, splits=SPLITS, seed=0, models=MODELS,
           onchart=True):
    """Long predictions table in the format of ``predicoes_cinco_modelos.parquet``.

    ``noise[model]`` is the std of ``y_pred - y_true``; ``origins`` origins per song.
    """
    noise = noise or {"atencao": 0.005, "sem_grafo": 0.05, "atencao_emb": 0.05,
                      "proposta": 0.05}
    rng = np.random.default_rng(seed)
    rows = []
    for split in splits:
        for chart in CHARTS:
            for k in KS:
                for s in range(n_songs):
                    for o in range(origins):
                        y = float(rng.uniform(0.0, 0.5))
                        for m in models:
                            rows.append({
                                "song_id": f"s{s}", "chart": chart, "origin_week": 100 + o,
                                "k": k, "split_regime": split, "modelo": m, "y_true": y,
                                "y_pred": y + float(rng.normal(0, noise[m])),
                                "onchart": onchart,
                            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ T6: compare_models

def test_contrasts_are_the_three_of_adr_0006():
    assert TREATED == "atencao"
    assert CONTRASTS == {"H1": "sem_grafo", "H2": "atencao_emb", "H3": "proposta"}


def test_eighteen_rows_per_split_and_holm_over_the_eighteen_together():
    t = compare_models(_preds(), B=B, seed=1)
    assert t.groupby("split_regime").size().to_dict() == {"current": 18, "pre_pandemia": 18}
    for _, grp in t.groupby("split_regime"):
        assert set(zip(grp.chart, grp.k, grp.hipotese)) == {
            (c, k, h) for c in CHARTS for k in KS for h in CONTRASTS
        }
        np.testing.assert_allclose(grp["p_holm"].to_numpy(),
                                   holm_correction(grp["p_value"].to_numpy()))


def test_difference_matches_paired_by_song_called_directly():
    p = _preds()
    t = compare_models(p, B=B, seed=7)
    row = t[(t.split_regime == "current") & (t.chart == "top200") & (t.k == 2)
            & (t.hipotese == "H2")].iloc[0]
    cell = p[(p.split_regime == "current") & (p.chart == "top200") & (p.k == 2)]
    a = cell[cell.modelo == "atencao"].sort_values(["song_id", "origin_week"])
    b = cell[cell.modelo == "atencao_emb"].sort_values(["song_id", "origin_week"])
    ref = paired_by_song(((a.y_true - a.y_pred) ** 2).to_numpy(),
                         ((b.y_true - b.y_pred) ** 2).to_numpy(),
                         a.song_id.to_numpy(), B=B, seed=7)
    assert row.contraste == "atencao_emb"
    for key in ("mean_diff", "ci_lo", "ci_hi", "p_value", "win_rate"):
        assert row[key] == pytest.approx(ref[key])
    assert row["n_songs"] == ref["n_songs"]


def test_origins_of_one_song_are_averaged_and_n_songs_counts_songs():
    t = compare_models(_preds(n_songs=30, origins=3), B=B, seed=1)
    assert (t["n_songs"] == 30).all()
    assert (t["n_origens"] == 90).all()


def test_vence_requires_whole_ci_below_zero_and_holm_p_below_005():
    assert verdict(-0.3, -0.1, 0.01) == "vence"
    assert verdict(-0.3, -0.1, 0.05) == "inconclusivo"   # significance fails
    assert verdict(-0.3, 0.1, 0.001) == "inconclusivo"   # CI crosses zero
    assert verdict(0.1, 0.3, 0.01) == "perde"
    assert verdict(0.1, 0.3, 0.2) == "inconclusivo"


def test_labels_of_the_table_follow_the_rule():
    noise = {"atencao": 0.005, "sem_grafo": 0.05, "atencao_emb": 0.05, "proposta": 0.0005}
    t = compare_models(_preds(noise), B=B, seed=1)
    assert (t[t.hipotese == "H1"].rotulo == "vence").all()
    assert (t[t.hipotese == "H3"].rotulo == "perde").all()
    assert t.rotulo.tolist() == [verdict(r.ci_lo, r.ci_hi, r.p_holm) for r in t.itertuples()]


def test_only_onchart_origins_enter():
    on = _preds(seed=3)
    # off-chart rows where the attention model is far worse: they must not change anything
    off = _preds({"atencao": 0.4, "sem_grafo": 0.001, "atencao_emb": 0.001, "proposta": 0.001},
                 seed=4, onchart=False)
    off["origin_week"] += 50
    t_on = compare_models(on, B=B, seed=1)
    t_all = compare_models(pd.concat([on, off], ignore_index=True), B=B, seed=1)
    pd.testing.assert_frame_equal(t_on, t_all)


@pytest.mark.parametrize("missing", ["atencao", "sem_grafo", "atencao_emb", "proposta"])
def test_missing_model_raises_naming_it(missing):
    p = _preds(splits=("current",), n_songs=10, origins=1)
    with pytest.raises(ValueError, match=missing):
        compare_models(p[p.modelo != missing], B=B, seed=1)


# ------------------------------------------------------------ T7: classify_outcome

def _labels(n_vence=0, n_perde=0):
    """Six cell labels: ``n_vence`` wins, ``n_perde`` losses, the rest inconclusive."""
    return ["vence"] * n_vence + ["perde"] * n_perde + ["inconclusivo"] * (6 - n_vence - n_perde)


def _table(split="current", *, h1=(0, 0), h2=(0, 0), h3=(0, 0)):
    """Synthetic compare_models output for one split; each hN is (wins, losses)."""
    cells = [(c, k) for c in CHARTS for k in KS]
    rows = []
    for hyp, (w, lo) in (("H1", h1), ("H2", h2), ("H3", h3)):
        for (c, k), lab in zip(cells, _labels(w, lo)):
            rows.append({"split_regime": split, "chart": c, "k": k, "hipotese": hyp,
                         "contraste": CONTRASTS[hyp], "rotulo": lab})
    return pd.DataFrame(rows)


def _outcome(t):
    return classify_outcome(t).set_index("split_regime")["desfecho"].to_dict()


def test_majority_is_four_of_six():
    assert MAJORITY == 4


@pytest.mark.parametrize("h1, h2, expected", [
    ((4, 0), (4, 0), "O1"),
    ((6, 0), (5, 0), "O1"),
    ((4, 0), (3, 0), "O2"),
    ((5, 0), (0, 2), "O2"),
    ((3, 0), (3, 0), "O3"),
    ((0, 3), (0, 0), "O3"),
    ((0, 4), (0, 0), "O4"),
])
def test_each_outcome_from_a_synthetic_table(h1, h2, expected):
    assert _outcome(_table(h1=h1, h2=h2)) == {"current": expected}


def test_three_of_six_wins_is_not_a_majority_and_four_is():
    assert _outcome(_table(h1=(3, 0), h2=(4, 0))) == {"current": "O3"}
    assert _outcome(_table(h1=(4, 0), h2=(4, 0))) == {"current": "O1"}
    assert _outcome(_table(h1=(4, 0), h2=(3, 0))) == {"current": "O2"}


def test_h2_without_h1_is_o3():
    assert _outcome(_table(h1=(2, 0), h2=(6, 0))) == {"current": "O3"}


def test_o4_takes_precedence_over_h2_majority():
    assert _outcome(_table(h1=(0, 4), h2=(5, 0))) == {"current": "O4"}


def test_h3_does_not_change_the_outcome():
    for h1, h2 in [((4, 0), (4, 0)), ((4, 0), (0, 0)), ((1, 0), (1, 0)), ((0, 5), (0, 0))]:
        assert _outcome(_table(h1=h1, h2=h2, h3=(6, 0))) == _outcome(
            _table(h1=h1, h2=h2, h3=(0, 6)))


def test_counts_are_reported_per_split():
    out = classify_outcome(_table(h1=(4, 1), h2=(2, 3), h3=(1, 0))).iloc[0]
    assert (out.vitorias_h1, out.derrotas_h1, out.vitorias_h2, out.vitorias_h3) == (4, 1, 2, 1)


def test_agreeing_splits_have_no_discordance():
    t = pd.concat([_table("current", h1=(4, 0)), _table("pre_pandemia", h1=(5, 0))])
    out = classify_outcome(t)
    assert out.desfecho.tolist() == ["O2", "O2"]
    assert not out.discordancia.any()
    assert (out.afirmacao_principal == "O2").all()


def test_disagreeing_splits_flag_discordance_and_restrict_the_main_claim():
    t = pd.concat([_table("current", h1=(4, 0), h2=(4, 0)), _table("pre_pandemia", h1=(1, 0))])
    out = classify_outcome(t).set_index("split_regime")
    assert out.loc["current", "desfecho"] == "O1"
    assert out.loc["pre_pandemia", "desfecho"] == "O3"
    assert out.discordancia.all()
    assert (out.afirmacao_principal == "restrita ao que se replica").all()
    assert "replica" in out.loc["current", "nota"]


# ------------------------------------------------------------ T8: per-seed sensitivity, budget

INVERTED = {"atencao": 0.2, "sem_grafo": 0.005, "atencao_emb": 0.005, "proposta": 0.005}


def _seeds(per_seed_noise):
    """One prediction table per seed (same origins), stacked with a ``seed`` column."""
    return pd.concat([_preds(noise, n_songs=30, origins=1, seed=0).assign(seed=s)
                      for s, noise in per_seed_noise.items()], ignore_index=True)


def test_three_equal_seeds_agree_three_of_three_in_every_cell():
    preds = _seeds({42: None, 43: None, 44: None})
    main = compare_models(preds[preds.seed == 42].drop(columns="seed"), B=B, seed=1)
    cells, outcomes = sensitivity_by_seed(preds, main, B=B, seed=1)
    assert len(cells) == 36
    assert (cells.seeds_concordam == 3).all()
    assert (cells.n_seeds == 3).all()
    assert (cells.rotulo_seed42 == cells.rotulo).all()
    assert (outcomes.seeds_concordam == 3).all()


def test_one_inverted_seed_agrees_two_of_three():
    preds = _seeds({42: None, 43: None, 44: INVERTED})
    main = compare_models(_preds(n_songs=30, origins=1, seed=0), B=B, seed=1)
    assert (main.rotulo == "vence").all()
    cells, _ = sensitivity_by_seed(preds, main, B=B, seed=1)
    assert (cells.seeds_concordam == 2).all()
    assert (cells.rotulo_seed44 == "perde").all()


def test_outcome_of_each_isolated_seed_next_to_the_outcome_of_the_mean():
    preds = _seeds({42: None, 43: None, 44: INVERTED})
    main = compare_models(_preds(n_songs=30, origins=1, seed=0), B=B, seed=1)
    _, outcomes = sensitivity_by_seed(preds, main, B=B, seed=1)
    out = outcomes.set_index("split_regime")
    assert out.loc["current", "desfecho_media"] == "O1"
    assert out.loc["current", "desfecho_seed42"] == "O1"
    assert out.loc["current", "desfecho_seed43"] == "O1"
    assert out.loc["current", "desfecho_seed44"] == "O4"
    assert out.loc["current", "seeds_concordam"] == 2
    assert set(out.index) == {"current", "pre_pandemia"}


def test_tuning_budget_constants_are_the_adr_ones():
    assert (N_CONFIGS_ATTENTION, N_CONFIGS_SAGE, SAGE_N_PARAMS) == (8, 24, 453_121)
    assert TIE_VAL_MSE == 0.000003   # tie rule of the tuning (spec, edge cases)


def test_budget_block_states_both_param_counts_and_the_tuning_budget():
    text = budget_block(431_250)
    assert "431.250" in text
    assert "453.121" in text
    assert "8 contra 24 configurações" in text
