"""Genre nodes as structural attributes instead of a learned table (ADR-0003)."""
from __future__ import annotations

import pandas as pd
import pytest

from music_diffusion_gnn.data.loaders import load_genre_network
from music_diffusion_gnn.graph.nodes import (
    GENRE_FEATURE_NAMES,
    build_genre_nodes,
    genre_attributes,
)
from music_diffusion_gnn.training.dataset import SPLIT_REGIMES


def test_genre_attributes_drops_self_loops_and_aggregates_years():
    # "funk" ↔ "pagode" appears in two years; the self-loop describes no relation.
    net = pd.DataFrame({
        "Source": ["funk", "funk", "funk", "samba"],
        "Target": ["pagode", "pagode", "funk", "pagode"],
        "Weight": [3.0, 2.0, 99.0, 4.0],
    })
    artists = pd.DataFrame({
        "artist_id": ["a1", "a2"],
        "genres_list": [["funk", "pagode"], ["funk"]],
    })

    attrs = genre_attributes(net, artists)

    assert attrs.loc["funk", "degree"] == 1          # pagode only, self-loop dropped
    assert attrs.loc["funk", "weighted_degree"] == 5.0  # 3 + 2 summed across years
    assert attrs.loc["pagode", "degree"] == 2        # funk + samba
    assert attrs.loc["funk", "n_artists"] == 2
    assert attrs.loc["samba", "n_artists"] == 0      # tagged by nobody in these years


def test_train_years_are_fully_contained_in_the_training_window():
    assert SPLIT_REGIMES["current"].train_years == [2017, 2018, 2019]
    assert SPLIT_REGIMES["pre_pandemia"].train_years == [2017, 2018]


def test_leaking_columns_are_never_loaded():
    net = load_genre_network([2017])
    assert list(net.columns) == ["Source", "Target", "Weight"]


@pytest.fixture(scope="module")
def artists_df():
    from music_diffusion_gnn.data.loaders import load_artists
    return load_artists()


def test_features_are_attributes_and_depend_on_the_training_years(artists_df):
    artist_id_map = {a: i for i, a in enumerate(sorted(artists_df["artist_id"])[:200])}

    x_current, id_map = build_genre_nodes(artists_df, artist_id_map, [2017, 2018, 2019])
    x_pre, id_map_pre = build_genre_nodes(artists_df, artist_id_map, [2017, 2018])

    assert x_current.shape == (len(id_map), len(GENRE_FEATURE_NAMES))
    assert id_map == id_map_pre                      # same universe, different attributes
    assert not (x_current == x_pre).all()            # the extra year moves the counts
    assert x_current.isfinite().all()
    assert set(x_current[:, 3].unique().tolist()) <= {0.0, 1.0}  # absent flag is binary
