"""Assertive (non-probabilistic) no-leakage proof for the recursive GNN rollout
(Mode 2, T8) — co-located per tasks.md's "Done when" checklist."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from music_diffusion_gnn.training.dataset import TEST_START_WEEK

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
    """Untrained model: zero-init head => Delta=0 exactly in eval() mode."""
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


def _pick_origin(weekly_df: pd.DataFrame, *, min_local_range: float = 1e-3):
    """Pick a (song_id, chart, week) origin within the test span, with at
    least 6 weeks of real history before it and real variation in y_week
    *specifically within [w, w+3]* (so leakage would be detectable: a flat
    region around w would pass the no-leakage check by coincidence)."""
    in_span = weekly_df[weekly_df["week"] >= TEST_START_WEEK + 4]
    grp = weekly_df.groupby(["song_id", "chart"])
    fsw = grp["week"].min()
    indexed = weekly_df.set_index(["song_id", "chart", "week"])["y_week"]

    for _, row in in_span.iterrows():
        key = (row["song_id"], row["chart"])
        w = int(row["week"])
        if w - fsw.get(key, w) < 6:
            continue
        local = [indexed.get((*key, w + step)) for step in (0, 1, 2, 3)]
        local = [v for v in local if v is not None]
        if len(local) >= 2 and (max(local) - min(local)) > min_local_range:
            return row["song_id"], row["chart"], w
    raise AssertionError("no suitable origin found for leakage test fixture")


def test_encode_weeks_never_called_with_week_above_origin(graph, weekly_df, fresh_model):
    """Spy on encode_weeks: assert every call's weeks are all <= origin."""
    from music_diffusion_gnn.evaluation.rollout import gnn_rollout_recursive

    song_id, chart, w = _pick_origin(weekly_df)
    seen_weeks: list[int] = []
    orig_encode = fresh_model.encode_weeks

    def _spy(g, weeks, **kw):
        seen_weeks.extend(weeks)
        return orig_encode(g, weeks, **kw)

    fresh_model.encode_weeks = _spy
    try:
        origins = pd.DataFrame({"song_id": [song_id], "chart": [chart], "week": [w]})
        out = gnn_rollout_recursive(fresh_model, graph, weekly_df, origins, W=4, ks=(1, 2, 4))
    finally:
        fresh_model.encode_weeks = orig_encode

    assert len(seen_weeks) > 0, "encode_weeks was never called — fixture origin likely filtered out"
    assert max(seen_weeks) <= w, f"encode_weeks called with week {max(seen_weeks)} > origin {w}"
    assert len(out) == 3


def test_recursive_chain_uses_only_predicted_values_not_real_future(graph, weekly_df, fresh_model):
    """With Delta=0 exactly (zero-init head), every horizon's anchor is the
    model's OWN k=1 prediction (= real y at the origin), never the real
    trajectory at w+1, w+2, w+3 — proving work_bank[>origin] never receives a
    real value during the rollout."""
    from music_diffusion_gnn.evaluation.rollout import gnn_rollout_recursive

    song_id, chart, w = _pick_origin(weekly_df)
    real = weekly_df[(weekly_df["song_id"] == song_id) & (weekly_df["chart"] == chart)].set_index("week")[
        "y_week"
    ]

    origins = pd.DataFrame({"song_id": [song_id], "chart": [chart], "week": [w]})
    out = gnn_rollout_recursive(fresh_model, graph, weekly_df, origins, W=4, ks=(1, 2, 4)).set_index("k")

    y_real_w = float(real.loc[w])
    for k in (1, 2, 4):
        assert out.loc[k, "y_pred"] == pytest.approx(y_real_w, abs=1e-6)

    # Not a coincidence: the real trajectory actually moves at w+1..w+3.
    diverges = any(abs(real.loc[w + step] - y_real_w) > 1e-6 for step in (1, 2, 3) if (w + step) in real.index)
    assert diverges, "real trajectory matched y(w) everywhere — pick a higher-variance fixture origin"


def test_origins_outside_test_span_are_dropped(graph, weekly_df, fresh_model):
    from music_diffusion_gnn.evaluation.rollout import gnn_rollout_recursive

    origins = pd.DataFrame(
        {"song_id": ["nonexistent"], "chart": ["top200"], "week": [TEST_START_WEEK - 1]}
    )
    out = gnn_rollout_recursive(fresh_model, graph, weekly_df, origins, W=4, ks=(1, 2, 4))
    assert out.empty
