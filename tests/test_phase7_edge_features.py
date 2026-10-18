"""Causal edge features of the attention encoder (Phase 7, ATT-01, ADR-0006)."""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from music_diffusion_gnn.graph.edge_features import causal_edge_features
from music_diffusion_gnn.graph.temporal import mask_until

COTRAJ = ("music", "cotrajectory", "music")

# Five cotrajectory edges: [days_together, avg_position_distance, chart, first_seen_week]
_COTRAJ_ATTR = torch.tensor([
    [10.0, 3.5, 0.0, 0.0],
    [20.0, 1.0, 1.0, 3.0],
    [7.0, 0.0, 0.0, 5.0],
    [9.0, 2.0, 1.0, 5.0],
    [30.0, 4.0, 1.0, 1.0],
])
# Hand-computed for week 5: [chart, log1p(5 - first_seen_week)]
_EXPECTED_WEEK5 = torch.tensor([
    [0.0, 1.7917595],   # log1p(5)
    [1.0, 1.0986123],   # log1p(2)
    [0.0, 0.0],         # log1p(0)
    [1.0, 0.0],         # log1p(0)
    [1.0, 1.6094379],   # log1p(4)
])


def _graph(cotraj_attr: torch.Tensor = _COTRAJ_ATTR) -> HeteroData:
    """Small graph with all five edge types and their real attribute layouts."""
    g = HeteroData()
    g["music"].x = torch.randn(6, 3)
    g["artist"].x = torch.randn(3, 2)
    g["genre"].x = torch.randn(2, 2)
    g["artist", "performs", "music"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    g["artist", "performs", "music"].edge_attr = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [0.0, 0.0, 4.0]]
    )
    g["artist", "has_genre", "genre"].edge_index = torch.tensor([[0, 1], [0, 1]])
    g["artist", "has_genre", "genre"].first_seen_week = torch.zeros(2, dtype=torch.long)
    g["genre", "rev_has_genre", "artist"].edge_index = torch.tensor([[0, 1], [0, 1]])
    g["genre", "rev_has_genre", "artist"].first_seen_week = torch.zeros(2, dtype=torch.long)
    g["genre", "cooccurs", "genre"].edge_index = torch.tensor([[0, 1], [1, 0]])
    g["genre", "cooccurs", "genre"].edge_attr = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
    g[COTRAJ].edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    g[COTRAJ].edge_attr = cotraj_attr.clone()
    return g


def test_two_columns_match_hand_table():
    out = causal_edge_features(_graph(), 5)
    feats = out[COTRAJ]
    assert feats.shape == (5, 2)
    torch.testing.assert_close(feats, _EXPECTED_WEEK5, atol=1e-6, rtol=0)


def test_raw_columns_do_not_reach_the_output():
    """AC3 at function level: days_together and avg_position_distance are ignored."""
    perturbed = _COTRAJ_ATTR.clone()
    perturbed[:, 0] = torch.tensor([999.0, -3.0, 0.0, 1e6, 42.0])
    perturbed[:, 1] = torch.tensor([-7.5, 123.0, 0.25, 8.0, -1e4])
    base = causal_edge_features(_graph(), 5)[COTRAJ]
    after = causal_edge_features(_graph(perturbed), 5)[COTRAJ]
    assert torch.equal(base, after)


def test_edge_newer_than_week_is_a_usage_error():
    """An unmasked snapshot (first_seen_week > week) raises instead of going silent."""
    with pytest.raises(ValueError):
        causal_edge_features(_graph(), 4)


def test_only_cotrajectory_carries_causal_features():
    out = causal_edge_features(_graph(), 5)
    assert set(out) == {COTRAJ}
    for et in (
        ("artist", "performs", "music"),
        ("artist", "has_genre", "genre"),
        ("genre", "rev_has_genre", "artist"),
        ("genre", "cooccurs", "genre"),
    ):
        assert et not in out


@pytest.mark.parametrize("week", [0, 1, 3, 5, 100, 260])
def test_age_is_never_negative_on_a_masked_snapshot(week):
    snap = mask_until(_graph(), week)
    feats = causal_edge_features(snap, week)[COTRAJ]
    assert feats.shape[0] == int((_COTRAJ_ATTR[:, -1] <= week).sum())
    assert (feats[:, 1] >= 0).all()
