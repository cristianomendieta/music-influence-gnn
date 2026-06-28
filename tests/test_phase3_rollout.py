"""Tests for the free GNN rollout (Mode 1, OQ1) — T7."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "processed" / "graph" / "hetero_full.pt"
TS_PATH = ROOT / "data" / "processed" / "timeseries.parquet"
NMAP_PATH = ROOT / "data" / "processed" / "graph" / "node_id_map.json"


@pytest.fixture(scope="module")
def graph():
    return torch.load(GRAPH_PATH, weights_only=False)


@pytest.fixture(scope="module")
def weekly_df():
    from music_diffusion_gnn.training.dataset import aggregate_weekly

    return aggregate_weekly(pd.read_parquet(TS_PATH))


@pytest.fixture(scope="module")
def pop_bank(graph, weekly_df):
    from music_diffusion_gnn.training.dataset import build_pop_bank

    return build_pop_bank(weekly_df, NMAP_PATH, n_music=graph["music"].num_nodes)


@pytest.fixture(scope="module")
def fresh_model(graph, pop_bank):
    """Untrained model: zero-init head => Delta=0 exactly in eval() mode.

    This lets tests assert *exact* propagation behaviour (ŷ = y_prev every
    step) without needing the slow trained checkpoint.
    """
    from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN

    model = MusicDiffusionGNN(
        graph.metadata(),
        n_genre=graph["genre"].num_nodes,
        hidden=16,
        layers=1,
        dropout=0.0,
        pop_bank=pop_bank,
    )
    model.eval()
    return model


def _pick_key(weekly_df: pd.DataFrame, *, min_span: int, max_span: int, min_std: float = 1e-4):
    """Pick a (song_id, chart) with a short span and real variation in y_week."""
    grp = weekly_df.groupby(["song_id", "chart"])
    spans = grp["week"].agg(["min", "max"])
    spans["span"] = spans["max"] - spans["min"] + 1
    stds = grp["y_week"].std()
    candidates = spans[(spans["span"] >= min_span) & (spans["span"] <= max_span)]
    candidates = candidates[stds.reindex(candidates.index).fillna(0.0) > min_std]
    assert len(candidates) > 0, f"no (song,chart) with span in [{min_span},{max_span}] and std>{min_std}"
    return candidates.index[0]


def test_rollout_columns_shapes_and_bank_restored(graph, weekly_df, fresh_model):
    from music_diffusion_gnn.evaluation.rollout import gnn_rollout_free

    song_id, chart = _pick_key(weekly_df, min_span=10, max_span=20)
    sub = weekly_df[(weekly_df["song_id"] == song_id) & (weekly_df["chart"] == chart)]

    bank_before = fresh_model.pop_bank.clone()
    out = gnn_rollout_free(fresh_model, graph, sub, W=4, seed_weeks=4, device="cpu")

    assert list(out.columns) == ["song_id", "chart", "week", "y_true", "y_pred"]
    assert len(out) > 0
    assert (out["y_pred"] >= 0).all() and (out["y_pred"] <= 0.5).all()
    # The model's pop_bank buffer must be restored to its original (real) values.
    assert torch.allclose(fresh_model.pop_bank, bank_before)


def test_seed_anchor_real_then_self_fed(graph, weekly_df, fresh_model):
    """First evaluated week anchors to the real seed value (persistence);
    later weeks propagate the model's OWN prediction, never the real
    trajectory — this is the OQ1 fairness fix."""
    from music_diffusion_gnn.evaluation.rollout import gnn_rollout_free

    song_id, chart = _pick_key(weekly_df, min_span=10, max_span=20)
    sub = weekly_df[(weekly_df["song_id"] == song_id) & (weekly_df["chart"] == chart)].sort_values("week")
    fsw = int(sub["week"].min())
    seed_weeks = 4
    eval_start = fsw + seed_weeks

    out = gnn_rollout_free(fresh_model, graph, sub, W=4, seed_weeks=seed_weeks, device="cpu").set_index("week")
    real = sub.set_index("week")["y_week"]

    # Untrained model: Delta=0 exactly => ŷ = clamp(y_prev, 0, 0.5) = y_prev.
    # First prediction's anchor (week eval_start-1) is still within the seed,
    # i.e. the real value.
    assert out.loc[eval_start, "y_pred"] == pytest.approx(float(real.loc[eval_start - 1]), abs=1e-6)

    later_weeks = [w for w in out.index if w > eval_start]
    assert len(later_weeks) > 0
    for w in later_weeks:
        # The chain just keeps copying its own first prediction forward,
        # because every step anchors to the PREVIOUS PREDICTION (work_bank),
        # not the real y(w-1).
        assert out.loc[w, "y_pred"] == pytest.approx(out.loc[eval_start, "y_pred"], abs=1e-6)

    # Prove this isn't a coincidence: the real trajectory actually moves
    # while the rollout stays frozen at the first prediction.
    diverges = any(abs(real.loc[w] - out.loc[w, "y_pred"]) > 1e-6 for w in later_weeks)
    assert diverges, "real trajectory matched the frozen rollout everywhere — pick a higher-variance fixture song"


def test_span_less_than_2_excluded(graph, weekly_df, fresh_model):
    from music_diffusion_gnn.evaluation.rollout import gnn_rollout_free

    song_id, chart = weekly_df.iloc[0][["song_id", "chart"]]
    one_row = weekly_df[(weekly_df["song_id"] == song_id) & (weekly_df["chart"] == chart)].sort_values("week").iloc[[0]]

    out = gnn_rollout_free(fresh_model, graph, one_row, W=4, seed_weeks=4, device="cpu")
    assert out.empty
    assert list(out.columns) == ["song_id", "chart", "week", "y_true", "y_pred"]


def test_short_span_seed_clamped(graph, weekly_df, fresh_model):
    """span <= W => seed = max(1, span-1); a span-3 song with W=4 yields
    exactly one evaluated week (the last one)."""
    from music_diffusion_gnn.evaluation.rollout import gnn_rollout_free

    song_id, chart = _pick_key(weekly_df, min_span=5, max_span=200)
    full = weekly_df[(weekly_df["song_id"] == song_id) & (weekly_df["chart"] == chart)].sort_values("week")
    three = full.iloc[:3]
    fsw, lsw = int(three["week"].min()), int(three["week"].max())
    assert lsw - fsw + 1 == 3, "weekly_df expected dense per (song,chart) span (S3)"

    out = gnn_rollout_free(fresh_model, graph, three, W=4, seed_weeks=4, device="cpu")
    assert list(out["week"]) == [lsw]


def test_saturation_rate():
    from music_diffusion_gnn.evaluation.rollout import _saturation_rate

    assert _saturation_rate([0.0, 0.5, 0.25, 0.1]) == pytest.approx(0.5)
    assert _saturation_rate([0.2, 0.3, 0.4]) == pytest.approx(0.0)
    assert _saturation_rate([]) == 0.0
