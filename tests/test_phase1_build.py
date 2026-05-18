"""Integration tests for build_hetero — validates C1-C7 on real data.

Marked with @pytest.mark.slow; run with:
    pytest tests/test_phase1_build.py -m slow -xvs
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_GRAPH = ROOT / "data" / "processed" / "graph"
SUBSET_PATH = ROOT / "data" / "processed" / "subset_ids.json"


@pytest.fixture(scope="module")
def hetero_graph():
    """Load or build the heterogeneous graph once for all tests in this module."""
    pt_path = PROCESSED_GRAPH / "hetero_full.pt"
    if pt_path.exists():
        return torch.load(pt_path, weights_only=False)
    # Build if not cached
    from music_diffusion_gnn.graph.build import build_hetero
    return build_hetero(PROCESSED_GRAPH)


@pytest.fixture(scope="module")
def id_map():
    map_path = PROCESSED_GRAPH / "node_id_map.json"
    if not map_path.exists():
        pytest.skip("node_id_map.json not found — run build first")
    with open(map_path) as f:
        return json.load(f)


@pytest.mark.slow
class TestBuildCounts:
    def test_c1_n_music(self, hetero_graph):
        # Tolerance ±100: actual universe is 6526 (top200 ∪ viral50-with-features); spec estimated 6469
        n = hetero_graph["music"].num_nodes
        assert abs(n - 6469) <= 100, f"C1 FAIL: n_music={n}"

    def test_c2_n_artist(self, hetero_graph):
        n = hetero_graph["artist"].num_nodes
        assert abs(n - 1701) <= 5, f"C2 FAIL: n_artist={n}"

    def test_c3_n_genre(self, hetero_graph):
        n = hetero_graph["genre"].num_nodes
        assert abs(n - 530) <= 10, f"C3 FAIL: n_genre={n}"


@pytest.mark.slow
class TestSubsetCoverage:
    def test_c4_subset_in_graph(self, hetero_graph, id_map):
        if not SUBSET_PATH.exists():
            pytest.skip("subset_ids.json not found")
        from music_diffusion_gnn.data.subset import load_subset
        subset_ids = load_subset(SUBSET_PATH)
        music_to_idx = id_map["music"]["spotify_id_to_idx"]
        missing = [s for s in subset_ids if s not in music_to_idx]
        assert len(missing) == 0, f"C4 FAIL: {len(missing)} subset songs missing: {missing[:5]}"

    def test_c5_subset_artists_reachable(self, hetero_graph, id_map):
        if not SUBSET_PATH.exists():
            pytest.skip("subset_ids.json not found")
        from music_diffusion_gnn.data.subset import load_subset
        subset_ids = load_subset(SUBSET_PATH)
        music_to_idx = id_map["music"]["spotify_id_to_idx"]
        perf_et = ("artist", "performs", "music")
        assert perf_et in [tuple(e) for e in hetero_graph.edge_types], "No performs edges"
        ei = hetero_graph[perf_et].edge_index
        music_with_edges = set(ei[1].tolist())
        subset_idxs = {music_to_idx[s] for s in subset_ids if s in music_to_idx}
        isolated = subset_idxs - music_with_edges
        assert len(isolated) == 0, f"C5 FAIL: {len(isolated)} subset music nodes isolated"


@pytest.mark.slow
class TestEdgeValidity:
    def test_c6_no_dangling_edges(self, hetero_graph):
        g = hetero_graph
        n_music = g["music"].num_nodes
        n_artist = g["artist"].num_nodes
        n_genre = g["genre"].num_nodes

        checks = [
            ("artist", "performs", "music", n_artist, n_music),
            ("artist", "has_genre", "genre", n_artist, n_genre),
            ("genre", "rev_has_genre", "artist", n_genre, n_artist),
            ("music", "cotrajectory", "music", n_music, n_music),
            ("genre", "cooccurs", "genre", n_genre, n_genre),
        ]
        for src_t, rel, dst_t, n_src, n_dst in checks:
            et = (src_t, rel, dst_t)
            if et not in [tuple(e) for e in g.edge_types]:
                continue
            ei = g[et].edge_index
            if ei.shape[1] == 0:
                continue
            assert ei[0].max() < n_src, f"C6 FAIL {et}: src idx {ei[0].max()} >= {n_src}"
            assert ei[1].max() < n_dst, f"C6 FAIL {et}: dst idx {ei[1].max()} >= {n_dst}"

    def test_c7_first_seen_week_range(self, hetero_graph):
        g = hetero_graph
        for et in g.edge_types:
            store = g[et]
            keys = set(store.keys())
            if "edge_attr" in keys and store.edge_attr is not None and store.edge_attr.shape[1] > 0:
                fsw = store.edge_attr[:, -1]
            elif "first_seen_week" in keys:
                fsw = store.first_seen_week.float()
            else:
                continue
            assert (fsw >= 0).all(), f"C7 FAIL {et}: fsw < 0, min={fsw.min()}"
            assert (fsw <= 260).all(), f"C7 FAIL {et}: fsw > 260, max={fsw.max()}"


@pytest.mark.slow
class TestIdMap:
    def test_id_map_structure(self, id_map):
        for ntype in ("music", "artist", "genre"):
            assert ntype in id_map
        assert "spotify_id_to_idx" in id_map["music"]
        assert "idx_to_spotify_id" in id_map["music"]
        assert "artist_id_to_idx" in id_map["artist"]
        assert "genre_name_to_idx" in id_map["genre"]

    def test_id_map_consistency(self, id_map):
        for ntype in ("music", "artist", "genre"):
            to_idx_key = [k for k in id_map[ntype] if k.endswith("_to_idx")][0]
            to_name_key = [k for k in id_map[ntype] if k.startswith("idx_to")][0]
            to_idx = id_map[ntype][to_idx_key]
            to_name = id_map[ntype][to_name_key]
            assert len(to_idx) == len(to_name), f"Mismatch in {ntype} id_map sizes"
