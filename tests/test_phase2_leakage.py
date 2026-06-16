"""C2: no temporal leakage — y(w) never sees edges or embeddings from week >= w."""
from __future__ import annotations

import json
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
    df = pd.read_parquet(TS_PATH)
    return aggregate_weekly(df)


@pytest.fixture(scope="module")
def small_samples(weekly_df):
    from music_diffusion_gnn.training.dataset import temporal_split, build_samples
    splits = temporal_split(weekly_df)
    # Use only first 20 train songs for speed
    songs = splits["train"]["song_id"].unique()[:20]
    sub = splits["train"][splits["train"]["song_id"].isin(songs)]
    return build_samples(sub, W=4, node_id_map_path=NMAP_PATH)


# ---------------------------------------------------------------------------
# C2.1 — window_weeks contains only weeks strictly < target_week
# ---------------------------------------------------------------------------

def test_no_future_in_window(small_samples):
    """No window week should be >= the target week (causal constraint)."""
    violations = 0
    for samp in small_samples:
        for wk, is_pad in zip(samp.window_weeks, samp.pad_mask):
            if not is_pad:
                if wk >= samp.target_week:
                    violations += 1
    assert violations == 0, f"{violations} samples have window_week >= target_week"


# ---------------------------------------------------------------------------
# C2.2 — mask_until(g, w) excludes edges with first_seen_week > w
# ---------------------------------------------------------------------------

def test_mask_until_excludes_future_edges(graph):
    """mask_until(g, w) must contain no edges with first_seen_week > w."""
    from music_diffusion_gnn.graph.temporal import mask_until

    for test_week in (0, 50, 130, 200, 260):
        snap = mask_until(graph, test_week)
        for et in snap.edge_types:
            store = snap[et]
            keys = set(store.keys())
            if "edge_attr" in keys and store.edge_attr is not None:
                fsw = store.edge_attr[:, -1]
                violating = (fsw > test_week).sum().item()
                assert violating == 0, (
                    f"edge type {et} has {violating} edges with "
                    f"first_seen_week > {test_week} in mask_until(g, {test_week})"
                )
            elif "first_seen_week" in keys:
                fsw = store.first_seen_week.float()
                violating = (fsw > test_week).sum().item()
                assert violating == 0, (
                    f"edge type {et} has {violating} edges with "
                    f"first_seen_week > {test_week} in mask_until(g, {test_week})"
                )


# ---------------------------------------------------------------------------
# C2.3 — encode_weeks called with w-1 for a sample targeting week w
# ---------------------------------------------------------------------------

def test_encode_weeks_uses_past_snapshots(graph, small_samples):
    """encode_weeks should only be called with weeks <= target_week - 1."""
    from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
    from music_diffusion_gnn.training.trainer import _distinct_window_weeks

    model = MusicDiffusionGNN(graph.metadata(), n_genre=graph["genre"].num_nodes, hidden=64, layers=2, dropout=0.0)

    # Take a small batch and verify weeks in bank are all < target_week
    batch = small_samples[:16]
    target_weeks = {s.target_week for s in batch}
    window_weeks = set(_distinct_window_weeks(batch))

    # All window weeks must be strictly less than every target week in the batch
    # (each sample guarantees wk < target_week; here we check batch level)
    for wk in window_weeks:
        # Find which sample(s) use this week and check wk < their target_week
        users = [s for s in batch if wk in s.window_weeks]
        for s in users:
            assert wk < s.target_week, (
                f"Window week {wk} >= target_week {s.target_week}"
            )

    # Actually call encode_weeks and confirm it only receives those weeks
    bank = model.encode_weeks(graph, list(window_weeks))
    assert set(bank.keys()) == window_weeks
