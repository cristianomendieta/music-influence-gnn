"""Choosing the encoder inside MusicDiffusionGNN (Phase 7, ATT-01, ADR-0006).

Synthetic graph of a few dozen nodes with the five edge types and the attribute
layouts of ``hetero_full``.
"""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from music_diffusion_gnn.graph.edge_features import causal_edge_features
from music_diffusion_gnn.graph.temporal import mask_until
from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
from music_diffusion_gnn.training.dataset import Sample

COTRAJ = ("music", "cotrajectory", "music")
PERFORMS = ("artist", "performs", "music")
N_MUSIC, N_WEEKS, W = 10, 16, 4
HIDDEN = 8

# (src, dst, days_together, avg_position_distance, chart, first_seen_week).
# The first two rows are a parallel pair: the same songs in both charts.
_COTRAJ_ROWS = [
    (0, 1, 10, 2.0, 0, 0), (0, 1, 12, 3.0, 1, 1), (1, 2, 8, 1.0, 0, 2),
    (2, 3, 9, 4.0, 1, 2), (3, 4, 15, 0.5, 0, 3), (4, 5, 7, 1.0, 1, 4),
    (5, 6, 20, 2.0, 0, 5), (6, 7, 11, 3.0, 1, 6), (7, 8, 13, 1.0, 0, 7),
    (8, 9, 9, 2.0, 1, 8), (9, 0, 10, 1.0, 0, 9), (1, 3, 7, 2.0, 1, 3),
    (2, 4, 8, 1.0, 0, 1), (5, 4, 9, 1.5, 1, 2),
]

# state_dict of the default (sage) model, hidden=8, layers=2, captured from the
# code *before* the encoder option existed. The sage path must not move.
SAGE_STATE_SHAPES = [
    ("pop_bank", (16, 10, 2)),
    ("encoder.convs.0.convs.<artist___performs___music>.lin_l.weight", (8, 3)),
    ("encoder.convs.0.convs.<artist___performs___music>.lin_l.bias", (8,)),
    ("encoder.convs.0.convs.<artist___performs___music>.lin_r.weight", (8, 6)),
    ("encoder.convs.0.convs.<artist___has_genre___genre>.lin_l.weight", (8, 3)),
    ("encoder.convs.0.convs.<artist___has_genre___genre>.lin_l.bias", (8,)),
    ("encoder.convs.0.convs.<artist___has_genre___genre>.lin_r.weight", (8, 2)),
    ("encoder.convs.0.convs.<genre___rev_has_genre___artist>.lin_l.weight", (8, 2)),
    ("encoder.convs.0.convs.<genre___rev_has_genre___artist>.lin_l.bias", (8,)),
    ("encoder.convs.0.convs.<genre___rev_has_genre___artist>.lin_r.weight", (8, 3)),
    ("encoder.convs.0.convs.<genre___cooccurs___genre>.lin_l.weight", (8, 2)),
    ("encoder.convs.0.convs.<genre___cooccurs___genre>.lin_l.bias", (8,)),
    ("encoder.convs.0.convs.<genre___cooccurs___genre>.lin_r.weight", (8, 2)),
    ("encoder.convs.0.convs.<music___cotrajectory___music>.lin_l.weight", (8, 6)),
    ("encoder.convs.0.convs.<music___cotrajectory___music>.lin_l.bias", (8,)),
    ("encoder.convs.0.convs.<music___cotrajectory___music>.lin_r.weight", (8, 6)),
    ("encoder.convs.1.convs.<artist___performs___music>.lin_l.weight", (8, 8)),
    ("encoder.convs.1.convs.<artist___performs___music>.lin_l.bias", (8,)),
    ("encoder.convs.1.convs.<artist___performs___music>.lin_r.weight", (8, 8)),
    ("encoder.convs.1.convs.<artist___has_genre___genre>.lin_l.weight", (8, 8)),
    ("encoder.convs.1.convs.<artist___has_genre___genre>.lin_l.bias", (8,)),
    ("encoder.convs.1.convs.<artist___has_genre___genre>.lin_r.weight", (8, 8)),
    ("encoder.convs.1.convs.<genre___rev_has_genre___artist>.lin_l.weight", (8, 8)),
    ("encoder.convs.1.convs.<genre___rev_has_genre___artist>.lin_l.bias", (8,)),
    ("encoder.convs.1.convs.<genre___rev_has_genre___artist>.lin_r.weight", (8, 8)),
    ("encoder.convs.1.convs.<genre___cooccurs___genre>.lin_l.weight", (8, 8)),
    ("encoder.convs.1.convs.<genre___cooccurs___genre>.lin_l.bias", (8,)),
    ("encoder.convs.1.convs.<genre___cooccurs___genre>.lin_r.weight", (8, 8)),
    ("encoder.convs.1.convs.<music___cotrajectory___music>.lin_l.weight", (8, 8)),
    ("encoder.convs.1.convs.<music___cotrajectory___music>.lin_l.bias", (8,)),
    ("encoder.convs.1.convs.<music___cotrajectory___music>.lin_r.weight", (8, 8)),
    ("head.gru.weight_ih_l0", (24, 8)),
    ("head.gru.weight_hh_l0", (24, 8)),
    ("head.gru.bias_ih_l0", (24,)),
    ("head.gru.bias_hh_l0", (24,)),
    ("head.mlp.1.weight", (8, 8)),
    ("head.mlp.1.bias", (8,)),
    ("head.mlp.4.weight", (1, 8)),
    ("head.mlp.4.bias", (1,)),
]


def _graph(seed: int = 0) -> HeteroData:
    gen = torch.Generator().manual_seed(seed)
    g = HeteroData()
    g["music"].x = torch.randn(N_MUSIC, 4, generator=gen)
    g["artist"].x = torch.randn(4, 3, generator=gen)
    g["genre"].x = torch.randn(3, 2, generator=gen)
    music = torch.arange(N_MUSIC)
    g[PERFORMS].edge_index = torch.stack([music % 4, music])
    g[PERFORMS].edge_attr = torch.stack(
        [torch.zeros(N_MUSIC), torch.zeros(N_MUSIC), music.float()], dim=1
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


def _pop_bank() -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    return torch.rand(N_WEEKS, N_MUSIC, 2, generator=gen) * 0.5


def _samples(target_week: int = 12) -> list[Sample]:
    window = list(range(target_week - W, target_week))
    return [
        Sample(song_idx=s, chart=s % 2, target_week=target_week, window_weeks=window,
               pad_mask=[False] * W, y=0.1)
        for s in range(N_MUSIC)
    ]


def _model(**kw) -> MusicDiffusionGNN:
    torch.manual_seed(0)
    m = MusicDiffusionGNN(_graph().metadata(), hidden=HIDDEN, layers=2, dropout=0.0,
                          pop_bank=_pop_bank(), **kw)
    m.eval()
    return m


def _predict(m: MusicDiffusionGNN, g: HeteroData, **kw) -> torch.Tensor:
    samples = _samples()
    with torch.no_grad():
        bank = m.encode_weeks(g, samples[0].window_weeks, **kw)
        return m.predict(bank, samples)


@pytest.mark.parametrize("kw", [{}, {"encoder": "sage"}])
def test_sage_state_dict_is_unchanged(kw):
    m = _model(**kw)
    _predict(m, _graph())  # materialise the lazy SAGE weights
    shapes = [(k, tuple(v.shape)) for k, v in m.state_dict().items()]
    assert shapes == SAGE_STATE_SHAPES


def test_gatv2_predicts_within_range():
    m = _model(encoder="gatv2", heads=2)
    y_hat = _predict(m, _graph())
    assert y_hat.shape == (N_MUSIC,)
    assert torch.isfinite(y_hat).all()
    assert (y_hat >= 0).all() and (y_hat <= 0.5).all()


def test_raw_edge_attr_columns_do_not_change_predictions():
    """AC3 end to end: days_together / avg_position_distance never reach the model."""
    g = _graph()
    perturbed = g.clone()
    perturbed[COTRAJ].edge_attr[:, 0] = torch.randn(len(_COTRAJ_ROWS)) * 1e3
    perturbed[COTRAJ].edge_attr[:, 1] = torch.randn(len(_COTRAJ_ROWS)) * 1e3
    m = _model(encoder="gatv2", heads=2)
    assert torch.equal(_predict(m, g), _predict(m, perturbed))


def test_subsampled_edges_keep_their_own_attributes():
    """With max_cotraj_edges, edge features follow the same perm as the edges."""
    g = _graph()
    week = 11
    snap = mask_until(g, week)
    feats = causal_edge_features(snap, week)[COTRAJ]
    full = {
        (int(s), int(d), float(c), float(a))
        for (s, d), (c, a) in zip(snap[COTRAJ].edge_index.t().tolist(), feats.tolist())
    }

    m = _model(encoder="gatv2", heads=2)
    seen = []
    orig_forward = m.encoder.forward

    def spy(x_dict, edge_index_dict, edge_attr_dict=None, **kw):
        seen.append((edge_index_dict[COTRAJ], edge_attr_dict[COTRAJ]))
        return orig_forward(x_dict, edge_index_dict, edge_attr_dict, **kw)

    m.encoder.forward = spy
    torch.manual_seed(3)
    with torch.no_grad():
        m.encode_weeks(g, [week], max_cotraj_edges=5)

    (edge_index, attr), = seen
    assert edge_index.shape == (2, 5)
    assert attr.shape == (5, 2)
    for (s, d), (c, a) in zip(edge_index.t().tolist(), attr.tolist()):
        assert (int(s), int(d), float(c), float(a)) in full


def test_unknown_encoder_lists_the_accepted_values():
    with pytest.raises(ValueError) as exc:
        _model(encoder="hgt")
    msg = str(exc.value)
    assert "sage" in msg and "gatv2" in msg and "hgt" in msg
