"""Config, training and checkpoint of the attention encoder (Phase 7, ATT-02, ATT-04).

Synthetic graph of a few dozen nodes with the five edge types and the attribute
layouts of ``hetero_full``.
"""
from __future__ import annotations

import io
import math
import re

import pytest
import torch
from torch_geometric.data import HeteroData

from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
from music_diffusion_gnn.training.dataset import Sample
from music_diffusion_gnn.training.trainer import (
    Config,
    TrainResult,
    check_param_budget,
    checkpoint_dict,
    model_from_checkpoint,
    train_one,
)

COTRAJ = ("music", "cotrajectory", "music")
PERFORMS = ("artist", "performs", "music")
N_MUSIC, N_WEEKS, W = 10, 16, 4

# (src, dst, days_together, avg_position_distance, chart, first_seen_week)
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


def _samples(target_week: int) -> list[Sample]:
    window = list(range(target_week - W, target_week))
    return [
        Sample(song_idx=s, chart=s % 2, target_week=target_week, window_weeks=window,
               pad_mask=[False] * W, y=0.1 + 0.02 * s)
        for s in range(N_MUSIC)
    ]


def _cfg(**kw) -> Config:
    return Config(W=W, hidden=8, layers=2, lr=1e-3, max_epochs=1, dropout=0.0, **kw)


# --- Config.__str__ ---------------------------------------------------------

def test_sage_config_string_is_unchanged():
    assert str(Config(W=12, hidden=128, layers=3, lr=5e-4)) == "W12_h128_l3_lr5e-04"
    assert str(Config(W=12, hidden=128, layers=3, lr=5e-4, encoder="sage")) == "W12_h128_l3_lr5e-04"


def test_config_string_tells_encoders_and_heads_apart():
    def gat(heads, ad):
        return str(Config(W=12, hidden=128, layers=3, lr=5e-4, encoder="gatv2",
                          heads=heads, attn_dropout=ad))
    assert gat(4, 0.2) == "W12_h128_l3_lr5e-04_gatv2_h4_ad0.2"
    assert gat(2, 0.0) == "W12_h128_l3_lr5e-04_gatv2_h2_ad0"
    assert len({gat(2, 0.0), gat(4, 0.0), gat(2, 0.2), gat(4, 0.2)}) == 4


# --- parameter budget (AC6) ---------------------------------------------------

@pytest.mark.parametrize("n", [453_121, 407_808, 498_433])
def test_budget_accepts_within_ten_percent(n):
    check_param_budget(n)


@pytest.mark.parametrize("n", [407_807, 498_434, 100_000])
def test_budget_rejects_outside_with_both_numbers(n):
    with pytest.raises(ValueError) as exc:
        check_param_budget(n)
    msg = str(exc.value)
    assert re.search(rf"\b{n}\b", msg) and re.search(r"\b453121\b", msg), msg


# --- checkpoint (training AC4) ----------------------------------------------

def test_checkpoint_carries_metadata_and_rebuilds_the_model():
    g = _graph()
    cfg = _cfg(encoder="gatv2", heads=2, attn_dropout=0.2, seed=43)
    torch.manual_seed(0)
    model = MusicDiffusionGNN(g.metadata(), hidden=cfg.hidden, layers=cfg.layers,
                              dropout=cfg.dropout, pop_bank=_pop_bank(), encoder=cfg.encoder,
                              heads=cfg.heads, attn_dropout=cfg.attn_dropout).eval()
    samples = _samples(12)
    with torch.no_grad():
        y_ref = model.predict(model.encode_weeks(g, samples[0].window_weeks), samples)
    result = TrainResult(best_state_dict=model.state_dict(), train_curve=[0.1], val_curve=[0.2],
                         val_mse=0.2, n_params=model.count_params(), elapsed_sec=1.0)

    ck = checkpoint_dict(cfg, result, variante="A-emb", split_regime="pre_pandemia",
                         graph_seed=43, graph_version="hetero_full_pre_pandemia.pt")
    assert {k: ck[k] for k in ("encoder", "heads", "attn_dropout", "variante", "split_regime",
                               "seed", "graph_seed", "n_params")} == {
        "encoder": "gatv2", "heads": 2, "attn_dropout": 0.2, "variante": "A-emb",
        "split_regime": "pre_pandemia", "seed": 43, "graph_seed": 43,
        "n_params": model.count_params(),
    }
    assert ck["graph_version"] == "hetero_full_pre_pandemia.pt"

    buf = io.BytesIO()
    torch.save(ck, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    rebuilt = model_from_checkpoint(loaded, g.metadata()).eval()
    assert rebuilt.encoder_name == "gatv2"
    with torch.no_grad():
        y = rebuilt.predict(rebuilt.encode_weeks(g, samples[0].window_weeks), samples)
    assert torch.equal(y, y_ref)


# --- train_one with the attention encoder ------------------------------------

def test_train_one_runs_one_epoch_with_gatv2():
    g = _graph()
    splits = {"train": _samples(10) + _samples(11), "val": _samples(13)}
    cfg = _cfg(encoder="gatv2", heads=2, attn_dropout=0.2)
    result = train_one(cfg, splits, g, pop_bank=_pop_bank())
    assert len(result.train_curve) == 1
    assert math.isfinite(result.val_mse)
    assert isinstance(result.n_params, int) and result.n_params > 0
    # The settings reached the model: GATv2 attention vectors with 2 heads × 4 channels.
    att = [v for k, v in result.best_state_dict.items() if k.endswith(".att")]
    assert att and all(tuple(v.shape) == (1, 2, 4) for v in att)
