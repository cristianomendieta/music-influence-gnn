"""T1: load_grid_best_model loads the BEST Phase 2 checkpoint end-to-end."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "processed" / "graph" / "hetero_full.pt"
TS_PATH    = ROOT / "data" / "processed" / "timeseries.parquet"
NMAP_PATH  = ROOT / "data" / "processed" / "graph" / "node_id_map.json"
CKPT_PATH  = ROOT / "results" / "phase2_experimentos_v2" / "grid_best_model.pt"


@pytest.fixture(scope="module")
def graph():
    return torch.load(GRAPH_PATH, weights_only=False)


@pytest.fixture(scope="module")
def pop_bank(graph):
    from music_diffusion_gnn.training.dataset import aggregate_weekly, build_pop_bank
    w = aggregate_weekly(pd.read_parquet(TS_PATH))
    return build_pop_bank(w, NMAP_PATH, n_music=graph["music"].num_nodes)


def test_load_grid_best_model(graph, pop_bank):
    from music_diffusion_gnn.evaluation.model_io import load_grid_best_model

    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    assert "W12_h128_l3" in ck["config_str"]

    model = load_grid_best_model(CKPT_PATH, graph, pop_bank, device="cpu")

    n = model.count_params()
    assert 50_000 <= n <= 500_000, f"param count {n} outside [50K, 500K]"


def test_missing_keys_are_only_pop_bank(graph, pop_bank):
    from music_diffusion_gnn.evaluation.model_io import load_grid_best_model
    from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN

    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = MusicDiffusionGNN(
        graph.metadata(),
        n_genre=graph["genre"].num_nodes,
        hidden=ck["hidden"],
        layers=ck["layers"],
        dropout=ck["dropout"],
        pop_bank=pop_bank,
    )
    sd = dict(ck["state_dict"])
    sd.pop("pop_bank", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    assert set(missing) <= {"pop_bank"}
    assert not unexpected
