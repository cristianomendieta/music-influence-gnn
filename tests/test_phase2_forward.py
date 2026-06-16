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
def train_samples():
    from music_diffusion_gnn.training.dataset import aggregate_weekly, temporal_split, build_samples
    df = pd.read_parquet(TS_PATH)
    w = aggregate_weekly(df)
    splits = temporal_split(w)
    songs = splits["train"]["song_id"].unique()[:10]
    sub = splits["train"][splits["train"]["song_id"].isin(songs)]
    return build_samples(sub, W=8, node_id_map_path=NMAP_PATH)


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
