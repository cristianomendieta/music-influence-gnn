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
    TREATED,
    compare_models,
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
