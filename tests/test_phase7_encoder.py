"""Per-relation GATv2 encoder with a root term (Phase 7, ATT-01, ATT-02, ADR-0006).

Everything runs on a synthetic graph of a few dozen nodes with the five edge
types and the attribute layouts of ``hetero_full``.
"""
from __future__ import annotations

import re

import pytest
import torch
from torch_geometric.data import HeteroData

from music_diffusion_gnn.graph.controls import strip_all_edges
from music_diffusion_gnn.graph.edge_features import causal_edge_features
from music_diffusion_gnn.graph.temporal import mask_until
from music_diffusion_gnn.models.attention_encoder import HeteroAttentionEncoder
from music_diffusion_gnn.models.encoder import HeteroSpatialEncoder

COTRAJ = ("music", "cotrajectory", "music")
PERFORMS = ("artist", "performs", "music")
WEEK = 9
HIDDEN, HEADS = 8, 2

# (src, dst, days_together, avg_position_distance, chart, first_seen_week).
# The first two rows are a parallel pair: the same songs in both charts.
_COTRAJ_ROWS = [
    (0, 1, 10, 2.0, 0, 0), (0, 1, 12, 3.0, 1, 1), (1, 2, 8, 1.0, 0, 2),
    (2, 3, 9, 4.0, 1, 2), (3, 4, 15, 0.5, 0, 3), (4, 5, 7, 1.0, 1, 4),
    (5, 6, 20, 2.0, 0, 5), (6, 7, 11, 3.0, 1, 6), (7, 8, 13, 1.0, 0, 7),
    (8, 9, 9, 2.0, 1, 8), (9, 0, 10, 1.0, 0, 9), (1, 3, 7, 2.0, 1, 3),
    (2, 4, 8, 1.0, 0, 1), (5, 4, 9, 1.5, 1, 2),
]


def _graph(seed: int = 0) -> HeteroData:
    gen = torch.Generator().manual_seed(seed)
    g = HeteroData()
    g["music"].x = torch.randn(10, 4, generator=gen)
    g["artist"].x = torch.randn(4, 3, generator=gen)
    g["genre"].x = torch.randn(3, 2, generator=gen)
    music = torch.arange(10)
    g[PERFORMS].edge_index = torch.stack([music % 4, music])
    g[PERFORMS].edge_attr = torch.stack(
        [torch.zeros(10), torch.zeros(10), music.float()], dim=1
    )
    artist = torch.arange(4)
    g["artist", "has_genre", "genre"].edge_index = torch.stack([artist, artist % 3])
    g["artist", "has_genre", "genre"].first_seen_week = torch.zeros(4, dtype=torch.long)
    g["genre", "rev_has_genre", "artist"].edge_index = torch.stack([artist % 3, artist])
    g["genre", "rev_has_genre", "artist"].first_seen_week = torch.zeros(4, dtype=torch.long)
    g["genre", "cooccurs", "genre"].edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    g["genre", "cooccurs", "genre"].edge_attr = torch.tensor(
        [[0.5, 0.0], [0.5, 0.0], [0.2, 0.0], [0.2, 0.0]]
    )
    rows = torch.tensor(_COTRAJ_ROWS, dtype=torch.float32)
    g[COTRAJ].edge_index = rows[:, :2].long().t().contiguous()
    g[COTRAJ].edge_attr = rows[:, 2:].contiguous()
    return g


def _encoder(g: HeteroData, layers: int = 2, **kw) -> HeteroAttentionEncoder:
    torch.manual_seed(0)
    enc = HeteroAttentionEncoder(g.metadata(), hidden=HIDDEN, layers=layers, heads=HEADS,
                                 dropout=0.0, **kw)
    enc.eval()
    return enc


def _encode(enc: HeteroAttentionEncoder, g: HeteroData, week: int = WEEK, **kw):
    snap = mask_until(g, week)
    return enc(snap.x_dict, snap.edge_index_dict, causal_edge_features(snap, week), **kw)


def _with_empty(g: HeteroData, *edge_types) -> HeteroData:
    out = g.clone()
    for et in edge_types:
        out[et].edge_index = out[et].edge_index[:, :0]
        if "edge_attr" in out[et]:
            out[et].edge_attr = out[et].edge_attr[:0]
        if "first_seen_week" in out[et]:
            out[et].first_seen_week = out[et].first_seen_week[:0]
    return out


def test_output_is_music_by_hidden():
    g = _graph()
    z = _encode(_encoder(g), g)
    assert z.shape == (g["music"].num_nodes, HIDDEN)


@pytest.mark.parametrize("empty", [COTRAJ, PERFORMS, ("genre", "cooccurs", "genre")])
def test_one_empty_relation_does_not_break_the_encoder(empty):
    """AC7: a relation without edges is omitted without error."""
    g = _with_empty(_graph(), empty)
    z = _encode(_encoder(g), g)
    assert z.shape == (g["music"].num_nodes, HIDDEN)
    assert z.abs().sum() > 0


def test_music_without_incoming_edges_keeps_the_root_term():
    """AC7: with no edge arriving at music, Z_music is exactly ReLU(W_root·x + b_root)."""
    g = _with_empty(_graph(), COTRAJ, PERFORMS)
    enc = _encoder(g, layers=1)
    with torch.no_grad():
        _encode(enc, g)  # materialise the lazy weights
        # A non-zero GATv2 bias would leak in if empty relations were not omitted.
        for conv in enc.convs[0].values():
            conv.bias.fill_(1.0)
        z = _encode(enc, g)
        expected = enc.roots[0]["music"](g["music"].x).relu()
    assert z.abs().sum() > 0
    torch.testing.assert_close(z, expected, atol=0, rtol=0)


def test_raw_edge_attr_columns_do_not_change_embeddings():
    """AC3 at encoder level: perturbing days_together / avg_position_distance is a no-op."""
    g = _graph()
    enc = _encoder(g)
    perturbed = g.clone()
    perturbed[COTRAJ].edge_attr[:, 0] = torch.randn(len(_COTRAJ_ROWS)) * 1e3
    perturbed[COTRAJ].edge_attr[:, 1] = torch.randn(len(_COTRAJ_ROWS)) * 1e3
    with torch.no_grad():
        assert torch.equal(_encode(enc, g), _encode(enc, perturbed))


def test_chart_attribute_reaches_the_attention():
    """AC2: the causal attributes are actually read (else AC3 would pass vacuously)."""
    g = _graph()
    enc = _encoder(g)
    flipped = g.clone()
    flipped[COTRAJ].edge_attr[:, 2] = 1.0 - flipped[COTRAJ].edge_attr[:, 2]
    with torch.no_grad():
        assert not torch.allclose(_encode(enc, g), _encode(enc, flipped))


def test_edgeless_graph_matches_sage_with_summed_root_weights():
    """AC4: with strip_all_edges, W_root,t = Σ_r lin_r and b_root,t = Σ_r lin_l.bias over
    the relations arriving at t reproduce the HeteroSpatialEncoder music embedding."""
    g = strip_all_edges(_graph())
    snap = mask_until(g, WEEK)
    torch.manual_seed(1)
    sage = HeteroSpatialEncoder(g.metadata(), hidden=HIDDEN, layers=2, dropout=0.0).eval()
    enc = _encoder(g, layers=2)
    with torch.no_grad():
        sage(snap.x_dict, snap.edge_index_dict)
        _encode(enc, g)
        for layer in range(2):
            for t, root in enc.roots[layer].items():
                arriving = [et for et in g.edge_types if et[2] == t]
                convs = sage.convs[layer].convs
                root.weight.copy_(sum(convs[et].lin_r.weight for et in arriving))
                root.bias.copy_(sum(convs[et].lin_l.bias for et in arriving))
            for conv in enc.convs[layer].values():
                conv.bias.normal_()  # empty relations must not add their bias either
        z_sage = sage(snap.x_dict, snap.edge_index_dict)
        z_att = _encode(enc, g)
    assert z_sage.abs().sum() > 0
    torch.testing.assert_close(z_att, z_sage, atol=1e-6, rtol=0)


def test_hidden_not_divisible_by_heads_is_rejected():
    with pytest.raises(ValueError) as exc:
        HeteroAttentionEncoder(_graph().metadata(), hidden=10, layers=1, heads=4)
    msg = str(exc.value)
    assert re.search(r"\b10\b", msg) and re.search(r"\b4\b", msg), msg
