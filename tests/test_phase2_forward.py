"""End-to-end forward smoke test: shape, range, param count (C3, C4)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "processed" / "graph" / "hetero_full.pt"
TS_PATH    = ROOT / "data" / "processed" / "timeseries.parquet"
NMAP_PATH  = ROOT / "data" / "processed" / "graph" / "node_id_map.json"


@pytest.fixture(scope="module")
def graph():
    return torch.load(GRAPH_PATH, weights_only=False)


@pytest.fixture(scope="module")
def weekly_df():
    from music_diffusion_gnn.training.dataset import aggregate_weekly
    return aggregate_weekly(pd.read_parquet(TS_PATH))


@pytest.fixture(scope="module")
def train_samples():
    from music_diffusion_gnn.training.dataset import aggregate_weekly, temporal_split, build_samples
    df = pd.read_parquet(TS_PATH)
    w = aggregate_weekly(df)
    splits = temporal_split(w)
    songs = splits["train"]["song_id"].unique()[:10]
    sub = splits["train"][splits["train"]["song_id"].isin(songs)]
    return build_samples(sub, W=8, node_id_map_path=NMAP_PATH)


@pytest.fixture(scope="module")
def pop_bank(graph, weekly_df):
    from music_diffusion_gnn.training.dataset import build_pop_bank
    return build_pop_bank(weekly_df, NMAP_PATH, n_music=graph["music"].num_nodes)


# ---------------------------------------------------------------------------
# C3: ŷ ∈ [0, 0.5]
# ---------------------------------------------------------------------------

def test_output_range(graph, train_samples):
    """Model output must be in [0, 0.5] for all batch sizes and configs."""
    from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
    from music_diffusion_gnn.training.trainer import _distinct_window_weeks

    model = MusicDiffusionGNN(graph.metadata(), n_genre=graph["genre"].num_nodes, hidden=64, layers=2, dropout=0.0)
    model.eval()

    batch = train_samples[:8]
    weeks = _distinct_window_weeks(batch)
    with torch.no_grad():
        bank = model.encode_weeks(graph, weeks)
        y_hat = model.predict(bank, batch)

    assert y_hat.shape == (8,), f"Expected (8,), got {y_hat.shape}"
    assert (y_hat >= 0).all(), f"Negative predictions: {y_hat.min().item()}"
    assert (y_hat <= 0.5).all(), f"Predictions > 0.5: {y_hat.max().item()}"


# ---------------------------------------------------------------------------
# C4: param count ∈ [50K, 500K]
# ---------------------------------------------------------------------------

def test_param_count(graph):
    """Largest grid config (hidden=128, layers=3) must have 50K–500K params."""
    from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN

    model = MusicDiffusionGNN(graph.metadata(), n_genre=graph["genre"].num_nodes, hidden=128, layers=3, dropout=0.2)
    # Warm-up to initialize lazy SAGE params
    from music_diffusion_gnn.graph.temporal import mask_until
    snap = mask_until(graph, 10)
    with torch.no_grad():
        _ = model.encoder(snap.x_dict, snap.edge_index_dict)

    n = model.count_params()
    assert 50_000 <= n <= 500_000, f"param count {n} outside [50K, 500K]"


# ---------------------------------------------------------------------------
# Integration: encode_weeks → predict pipeline
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline(graph, train_samples):
    """encode_weeks + predict runs without error and produces correct shapes."""
    from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
    from music_diffusion_gnn.training.trainer import _distinct_window_weeks

    model = MusicDiffusionGNN(graph.metadata(), n_genre=graph["genre"].num_nodes, hidden=64, layers=2, dropout=0.0)
    model.eval()

    # Use different batch sizes
    for B in (1, 4, 8):
        batch = train_samples[:B]
        weeks = _distinct_window_weeks(batch)
        if not weeks:
            continue
        with torch.no_grad():
            bank = model.encode_weeks(graph, weeks)
            y_hat = model.predict(bank, batch)
        assert y_hat.shape == (B,), f"B={B}: shape={y_hat.shape}"
        assert (y_hat >= 0).all() and (y_hat <= 0.5).all()


# ---------------------------------------------------------------------------
# R1: popularity injection + residual head
# ---------------------------------------------------------------------------

def test_pop_injection_forward_runs(graph, train_samples, pop_bank):
    """With pop_bank, encode_weeks (with +2 node-feature channels) + predict run
    and stay in range."""
    from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
    from music_diffusion_gnn.training.trainer import _distinct_window_weeks

    model = MusicDiffusionGNN(graph.metadata(), n_genre=graph["genre"].num_nodes,
                              hidden=64, layers=2, dropout=0.0, pop_bank=pop_bank)
    model.eval()
    batch = train_samples[:8]
    weeks = _distinct_window_weeks(batch)
    with torch.no_grad():
        bank = model.encode_weeks(graph, weeks)
        y_hat = model.predict(bank, batch)
    assert y_hat.shape == (8,)
    assert (y_hat >= 0).all() and (y_hat <= 0.5).all()


def test_residual_starts_at_persistence(graph, train_samples, pop_bank):
    """R1-D2/D3: with the zero-init head, Δ=0 → ŷ must equal the persistence
    value y_prev = pop_bank[w-1, song, chart] for every sample."""
    from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
    from music_diffusion_gnn.training.trainer import _distinct_window_weeks

    model = MusicDiffusionGNN(graph.metadata(), n_genre=graph["genre"].num_nodes,
                              hidden=64, layers=2, dropout=0.0, pop_bank=pop_bank)
    model.eval()
    batch = train_samples[:16]
    weeks = _distinct_window_weeks(batch)
    with torch.no_grad():
        bank = model.encode_weeks(graph, weeks)
        y_hat = model.predict(bank, batch)

    expected = torch.tensor(
        [pop_bank[s.target_week - 1, s.song_idx, s.chart].item() for s in batch]
    )
    assert torch.allclose(y_hat, expected, atol=1e-6), (
        "zero-init residual model must reproduce persistence exactly; "
        f"max diff {(y_hat - expected).abs().max().item():.2e}"
    )
