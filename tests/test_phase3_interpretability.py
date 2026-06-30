"""Smoke tests for P2 interpretability analyses — T10."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "processed" / "graph" / "hetero_full.pt"
TS_PATH = ROOT / "data" / "processed" / "timeseries.parquet"
NMAP_PATH = ROOT / "data" / "processed" / "graph" / "node_id_map.json"

_SCHEMA = ["analysis", "component", "delta_rmse", "regime"]


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


@pytest.fixture(scope="module")
def tiny_val_samples(graph, weekly_df):
    """3 manually-constructed Samples for one song — avoids expensive build_samples
    over the full weekly_df, since smoke tests only need structural correctness."""
    from music_diffusion_gnn.training.dataset import Sample

    with open(NMAP_PATH) as f:
        nmap = json.load(f)
    spotify_to_idx = nmap["music"]["spotify_id_to_idx"]

    # Pick first song that exists in weekly_df and node_id_map
    wdf = weekly_df[weekly_df["song_id"].isin(spotify_to_idx)]
    grp = wdf.groupby(["song_id", "chart"])["week"]
    spans = (grp.max() - grp.min() + 1)
    key = spans[spans >= 6].index[0]
    song_id, chart = key
    song_idx = spotify_to_idx[song_id]
    chart_code = 0 if chart == "viral50" else 1
    fsw = int(wdf[(wdf["song_id"] == song_id) & (wdf["chart"] == chart)]["week"].min())
    W = 4
    target_week = fsw + W + 1  # a week with enough history

    samples = []
    for k in range(3):
        tw = target_week + k
        window = [tw - j for j in range(W, 0, -1)]
        window_weeks = [wk if wk >= fsw else -1 for wk in window]
        pad_mask = [wk == -1 for wk in window_weeks]
        samples.append(
            Sample(
                song_idx=song_idx,
                chart=chart_code,
                target_week=tw,
                window_weeks=window_weeks,
                pad_mask=pad_mask,
                y=0.1,
            )
        )
    return samples


def test_edge_type_ablation_smoke_and_no_mutation(graph, tiny_val_samples, fresh_model):
    from music_diffusion_gnn.evaluation.interpretability import edge_type_ablation

    edge_types_before = list(graph.edge_types)
    edge_index_snapshots = {et: graph[et].edge_index.clone() for et in edge_types_before}

    out = edge_type_ablation(fresh_model, graph, tiny_val_samples)

    assert list(out.columns) == _SCHEMA
    assert len(out) == len(edge_types_before)
    assert set(out["analysis"]) == {"edge_type_ablation"}
    assert set(out["regime"]) == {"all"}
    assert not out["delta_rmse"].isna().any()
    assert set(out["component"]) == {str(et) for et in edge_types_before}

    # Original g must be provably unmutated.
    assert graph.edge_types == edge_types_before
    for et in edge_types_before:
        assert torch.equal(graph[et].edge_index, edge_index_snapshots[et])


def test_feature_group_permutation_smoke_and_no_mutation(graph, tiny_val_samples, fresh_model):
    from music_diffusion_gnn.evaluation.interpretability import feature_group_permutation

    x_before = graph["music"].x.clone()
    n_cols = graph["music"].x.shape[1]
    assert n_cols >= 2
    groups = {"first_two_cols": [0, 1]}

    out = feature_group_permutation(fresh_model, graph, tiny_val_samples, groups)

    assert list(out.columns) == _SCHEMA
    assert len(out) == 1
    assert out.iloc[0]["analysis"] == "feature_group_permutation"
    assert out.iloc[0]["component"] == "first_two_cols"
    assert out.iloc[0]["regime"] == "all"
    assert not pd.isna(out.iloc[0]["delta_rmse"])

    # Original g["music"].x must be provably unmutated.
    assert torch.equal(graph["music"].x, x_before)


def test_population_analogs_columns_and_triangular_bump():
    from music_diffusion_gnn.evaluation.interpretability import population_analogs

    curves_df = pd.DataFrame({
        "song_id": ["s1"] * 5,
        "chart": ["viral50"] * 5,
        "week": [0, 1, 2, 3, 4],
        "y_pred_gnn": [0.1, 0.2, 0.3, 0.2, 0.1],
    })

    out = population_analogs(curves_df)
    assert list(out.columns) == _SCHEMA
    assert set(out["analysis"]) == {"population_analogs"}
    assert len(out) == 3  # beta, gamma, r0

    by_component = out.set_index("component")["delta_rmse"]
    assert by_component["s1|viral50|beta"] == pytest.approx(0.1, abs=1e-9)
    assert by_component["s1|viral50|gamma"] == pytest.approx(0.1, abs=1e-9)
    assert by_component["s1|viral50|r0"] == pytest.approx(1.0, abs=1e-9)
    assert (out["regime"] == "viral50").all()


def test_population_analogs_monotonic_falling_peak_at_start():
    from music_diffusion_gnn.evaluation.interpretability import population_analogs

    curves_df = pd.DataFrame({
        "song_id": ["s2"] * 4,
        "chart": ["top200"] * 4,
        "week": [0, 1, 2, 3],
        "y_pred_gnn": [0.4, 0.3, 0.2, 0.1],
    })

    out = population_analogs(curves_df).set_index("component")["delta_rmse"]
    assert out["s2|top200|beta"] == 0.0
    assert out["s2|top200|gamma"] == pytest.approx(0.1, abs=1e-9)
    assert out["s2|top200|r0"] == pytest.approx(0.0, abs=1e-9)


def test_population_analogs_monotonic_rising_peak_at_end():
    from music_diffusion_gnn.evaluation.interpretability import population_analogs

    curves_df = pd.DataFrame({
        "song_id": ["s3"] * 4,
        "chart": ["top200"] * 4,
        "week": [0, 1, 2, 3],
        "y_pred_gnn": [0.1, 0.2, 0.3, 0.4],
    })

    out = population_analogs(curves_df).set_index("component")["delta_rmse"]
    assert out["s3|top200|beta"] == pytest.approx(0.1, abs=1e-9)
    assert out["s3|top200|gamma"] == 0.0
    assert np.isnan(out["s3|top200|r0"])
