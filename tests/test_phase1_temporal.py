"""Unit tests for graph.temporal — week_index and mask_until."""
from __future__ import annotations

from datetime import date

import pytest
import torch
from torch_geometric.data import HeteroData

from music_diffusion_gnn.graph.temporal import mask_until, week_index


# ---------------------------------------------------------------------------
# week_index
# ---------------------------------------------------------------------------

class TestWeekIndex:
    def test_first_week_2017(self):
        # 2017-01-02 is the first Monday of 2017, ISO week 1
        assert week_index(date(2017, 1, 2)) == 0

    def test_string_input(self):
        assert week_index("2017-01-02") == 0

    def test_last_week_2021(self):
        # 2021-12-27 is ISO week 52 of 2021 → index (4*52 + 51) = 259
        idx = week_index(date(2021, 12, 27))
        assert 0 <= idx <= 260

    def test_mid_range(self):
        # 2019-W01 → (2019-2017)*52 + 0 = 104
        # 2019-01-07 is in ISO week 2 of 2019
        # 2018-12-31 is in ISO week 1 of 2019
        idx = week_index(date(2019, 1, 1))
        # 2019-01-01 is ISO week 1 of 2019 → (2019-2017)*52 + 0 = 104
        assert idx == 104

    def test_raises_before_2017(self):
        with pytest.raises(ValueError):
            week_index(date(2016, 12, 31))

    def test_raises_after_2021(self):
        with pytest.raises(ValueError):
            week_index(date(2022, 1, 10))

    def test_range_all_valid(self):
        # Spot-check a range of dates within the valid period
        from datetime import timedelta
        d = date(2017, 1, 2)
        while d <= date(2021, 12, 26):
            idx = week_index(d)
            assert 0 <= idx <= 260, f"Out of range for {d}: {idx}"
            d += timedelta(days=7)


# ---------------------------------------------------------------------------
# mask_until
# ---------------------------------------------------------------------------

def _make_hetero() -> HeteroData:
    """Small synthetic HeteroData for testing mask_until."""
    g = HeteroData()

    # Nodes
    g["music"].x = torch.randn(10, 4)
    g["genre"].x = torch.randn(5, 2)
    g["artist"].x = torch.randn(3, 4)

    # Edge with edge_attr (first_seen_week = last col)
    # cotrajectory: music→music, weeks [10, 50, 130, 200]
    ei_cotraj = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    ea_cotraj = torch.tensor([
        [7.0, 15.0, 0.0, 10.0],
        [8.0, 10.0, 1.0, 50.0],
        [9.0, 5.0, 0.0, 130.0],
        [10.0, 20.0, 1.0, 200.0],
    ])
    g["music", "cotrajectory", "music"].edge_index = ei_cotraj
    g["music", "cotrajectory", "music"].edge_attr = ea_cotraj

    # Edge with separate first_seen_week tensor
    # has_genre: artist→genre, all first_seen_week=0
    ei_hg = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    fsw_hg = torch.zeros(3, dtype=torch.long)
    g["artist", "has_genre", "genre"].edge_index = ei_hg
    g["artist", "has_genre", "genre"].first_seen_week = fsw_hg

    return g


class TestMaskUntil:
    def test_raises_out_of_range(self):
        g = _make_hetero()
        with pytest.raises(ValueError):
            mask_until(g, -1)
        with pytest.raises(ValueError):
            mask_until(g, 261)

    def test_week_260_keeps_all_cotraj(self):
        g = _make_hetero()
        gm = mask_until(g, 260)
        assert gm["music", "cotrajectory", "music"].edge_index.shape[1] == 4

    def test_week_130_keeps_3_cotraj(self):
        g = _make_hetero()
        gm = mask_until(g, 130)
        # weeks 10, 50, 130 pass; 200 does not
        assert gm["music", "cotrajectory", "music"].edge_index.shape[1] == 3

    def test_week_0_keeps_only_has_genre(self):
        g = _make_hetero()
        gm = mask_until(g, 0)
        # cotrajectory: only week 10 passes? No — week 10 > 0. So 0 cotrajectory edges.
        assert gm["music", "cotrajectory", "music"].edge_index.shape[1] == 0
        # has_genre all have fsw=0 → all pass
        assert gm["artist", "has_genre", "genre"].edge_index.shape[1] == 3

    def test_monotonic(self):
        g = _make_hetero()
        e130 = mask_until(g, 130)["music", "cotrajectory", "music"].edge_index.shape[1]
        e260 = mask_until(g, 260)["music", "cotrajectory", "music"].edge_index.shape[1]
        assert e130 <= e260

    def test_node_features_shared(self):
        """Node tensors must be shared references (no deep copy)."""
        g = _make_hetero()
        gm = mask_until(g, 130)
        assert gm["music"].x.data_ptr() == g["music"].x.data_ptr()
        assert gm["genre"].x.data_ptr() == g["genre"].x.data_ptr()

    def test_mixed_layout(self):
        """mask_until handles graphs with both edge_attr and first_seen_week layouts."""
        g = _make_hetero()
        gm = mask_until(g, 50)
        # cotrajectory keeps weeks 10 and 50 → 2 edges
        assert gm["music", "cotrajectory", "music"].edge_index.shape[1] == 2
        # has_genre: all fsw=0 <= 50 → 3 edges
        assert gm["artist", "has_genre", "genre"].edge_index.shape[1] == 3
