"""Parameter budget at the real feature widths (Phase 7, ATT-02 AC6, ADR-0006).

ADR-0006 asks the attention model to stay within ±10% of the 453.121
parameters of the SAGE proposal. The count only depends on the layer widths,
so a synthetic graph with the real input widths (music 12 raw columns + 2
popularity channels, artist 1, genre 4) and the five real edge types gives
the same count as ``hetero_full``. The expected numbers are the ADR's:

- SAGE, ``hidden=128``, ``layers=3``: 453.121 (the reference itself);
- GATv2 with the root term, ``hidden=128``: 561.537 (+23,9%), which is why
  the 2026-10-18 amendment moves the attention encoder to ``hidden=112``;
- GATv2, ``hidden=112``: 432.209 (-4,6%), inside the band, the same for
  ``heads`` 2 and 4.
"""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
from music_diffusion_gnn.training.trainer import check_param_budget

COTRAJ = ("music", "cotrajectory", "music")
PERFORMS = ("artist", "performs", "music")
N_MUSIC, N_ARTIST, N_GENRE, N_WEEKS, W = 10, 4, 3, 16, 4
MUSIC_RAW, ARTIST_DIM, GENRE_DIM = 12, 1, 4  # real widths; pop_bank adds 2 to music

# (src, dst, days_together, avg_position_distance, chart, first_seen_week)
_COTRAJ_ROWS = [
    (0, 1, 10, 2.0, 0, 0), (0, 1, 12, 3.0, 1, 1), (1, 2, 8, 1.0, 0, 2),
    (2, 3, 9, 4.0, 1, 2), (3, 4, 15, 0.5, 0, 3), (4, 5, 7, 1.0, 1, 4),
    (5, 6, 20, 2.0, 0, 5), (6, 7, 11, 3.0, 1, 6), (7, 8, 13, 1.0, 0, 7),
    (8, 9, 9, 2.0, 1, 8), (9, 0, 10, 1.0, 0, 9), (1, 3, 7, 2.0, 1, 3),
    (2, 4, 8, 1.0, 0, 1), (5, 4, 9, 1.5, 1, 2),
]

SAGE_N_PARAMS = 453_121          # ADR-0006: the SAGE proposal
GATV2_H128_N_PARAMS = 561_537    # ADR-0006 amendment of 2026-10-18: outside the band
GATV2_H112_N_PARAMS = 432_209    # ADR-0006 amendment of 2026-10-18: -4,6%, inside
ATTENTION_HIDDEN = 112           # ADR-0006 amendment of 2026-10-18


def _graph() -> HeteroData:
    gen = torch.Generator().manual_seed(0)
    g = HeteroData()
    g["music"].x = torch.randn(N_MUSIC, MUSIC_RAW, generator=gen)
    g["artist"].x = torch.randn(N_ARTIST, ARTIST_DIM, generator=gen)
    g["genre"].x = torch.randn(N_GENRE, GENRE_DIM, generator=gen)
    music = torch.arange(N_MUSIC)
    g[PERFORMS].edge_index = torch.stack([music % N_ARTIST, music])
    g[PERFORMS].edge_attr = torch.stack(
        [torch.zeros(N_MUSIC), torch.zeros(N_MUSIC), music.float()], dim=1
    )
    artist = torch.arange(N_ARTIST)
    g["artist", "has_genre", "genre"].edge_index = torch.stack([artist, artist % N_GENRE])
    g["artist", "has_genre", "genre"].first_seen_week = torch.zeros(N_ARTIST, dtype=torch.long)
    g["genre", "rev_has_genre", "artist"].edge_index = torch.stack([artist % N_GENRE, artist])
    g["genre", "rev_has_genre", "artist"].first_seen_week = torch.zeros(N_ARTIST, dtype=torch.long)
    g["genre", "cooccurs", "genre"].edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    g["genre", "cooccurs", "genre"].edge_attr = torch.tensor(
        [[0.5, 0.0], [0.5, 0.0], [0.2, 0.0], [0.2, 0.0]]
    )
    rows = torch.tensor(_COTRAJ_ROWS, dtype=torch.float32)
    g[COTRAJ].edge_index = rows[:, :2].long().t().contiguous()
    g[COTRAJ].edge_attr = rows[:, 2:].contiguous()
    return g


def _count(**kw) -> int:
    g = _graph()
    pop_bank = torch.rand(N_WEEKS, N_MUSIC, 2, generator=torch.Generator().manual_seed(1)) * 0.5
    m = MusicDiffusionGNN(g.metadata(), layers=3, dropout=0.0, pop_bank=pop_bank, **kw)
    m.eval()
    with torch.no_grad():
        m.encode_weeks(g, [W])  # materialise the lazy weights
    return m.count_params()


def test_sage_control_has_exactly_the_reference_count():
    # Confirms the synthetic widths reproduce the real graph's count.
    assert _count(hidden=128, encoder="sage") == SAGE_N_PARAMS


def test_attention_at_amended_hidden_fits_the_budget_for_both_heads():
    counts = {h: _count(hidden=ATTENTION_HIDDEN, encoder="gatv2", heads=h) for h in (2, 4)}
    for n in counts.values():
        check_param_budget(n)
    assert counts[2] == counts[4] == GATV2_H112_N_PARAMS, counts


@pytest.mark.parametrize("heads", [2, 4])
def test_attention_at_hidden_128_is_outside_the_budget(heads):
    # Why the amendment exists: hidden=128 with the root term overshoots by 23,9%.
    n = _count(hidden=128, encoder="gatv2", heads=heads)
    assert n == GATV2_H128_N_PARAMS
    with pytest.raises(ValueError):
        check_param_budget(n)
