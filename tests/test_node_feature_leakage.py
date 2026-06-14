"""P0 regression: node features must be static metadata, free of temporal leakage.

``mask_until`` filters *edges* only and reuses node feature tensors unchanged at
every weekly snapshot. Any node feature derived from a song's / artist's chart
activity over the full 2017-2021 window therefore leaks future (and test-period)
information into every snapshot. These tests pin the invariant that node features
do NOT depend on chart trajectory — they catch a regression if a full-series
aggregate (e.g. dias_no_chart, total_streams, num_hits) is reintroduced.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from music_diffusion_gnn.data.loaders import load_artists, load_charts, load_songs
from music_diffusion_gnn.graph.nodes import build_artist_nodes, build_music_nodes


@pytest.fixture(scope="module")
def raw():
    charts = load_charts().copy()
    charts["date"] = pd.to_datetime(charts["date"])
    songs = load_songs()
    artists = load_artists()
    return charts, songs, artists


# ---------------------------------------------------------------------------
# Dimensionality — leaky columns removed
# ---------------------------------------------------------------------------

def test_music_feature_dim_is_12(raw):
    charts, songs, _ = raw
    x, _ = build_music_nodes(charts, songs, pd.DataFrame())
    assert x.shape[1] == 12, (
        f"expected 12 static music features, got {x.shape[1]} — a full-series "
        f"chart aggregate (popularity/total_streams/dias_no_chart) may be back"
    )


def test_artist_feature_dim_is_1(raw):
    charts, songs, artists = raw
    _, music_map = build_music_nodes(charts, songs, pd.DataFrame())
    x, _ = build_artist_nodes(artists, music_map, charts, songs)
    assert x.shape[1] == 1, (
        f"expected 1 static artist feature, got {x.shape[1]} — a full-series "
        f"chart count (num_hits/num_collab_hits/anos_no_chart) may be back"
    )


# ---------------------------------------------------------------------------
# Behavioural invariant — features ignore (future) chart activity
# ---------------------------------------------------------------------------

def test_music_features_invariant_to_future_chart_activity(raw):
    """Injecting future chart appearances for an existing song must NOT change
    its node features. If any full-series chart aggregate were present, the
    song's vector (and the z-score stats) would shift and this would fail."""
    charts, songs, _ = raw
    x1, map1 = build_music_nodes(charts, songs, pd.DataFrame())

    sid = next(iter(map1))  # a song already in the universe
    last = charts["date"].max()
    extra = pd.DataFrame({
        "song_id": [sid] * 30,
        "date": [last + pd.Timedelta(days=i + 1) for i in range(30)],
        "rank": [1] * 30,
        "chart": ["top200"] * 30,
    })
    charts2 = pd.concat([charts, extra], ignore_index=True)
    x2, map2 = build_music_nodes(charts2, songs, pd.DataFrame())

    assert torch.allclose(x1[map1[sid]], x2[map2[sid]]), (
        "music node features changed after adding future chart activity — "
        "a full-series chart aggregate is leaking into node features"
    )
