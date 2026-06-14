"""Node builders for music, artist and genre node types."""
from __future__ import annotations

import ast
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

ACOUSTIC_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "valence", "tempo",
]

# song_type categorical encoding (deterministic order)
_SONG_TYPE_MAP: dict[str, int] = {
    "Solo": 0,
    "Collaboration": 1,
    "Feature": 2,
}


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Column-wise z-score; columns with zero std become 0."""
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (arr - mean) / std


def _parse_list_col(series: pd.Series) -> pd.Series:
    """Parse string-encoded Python lists in a Series."""
    return series.apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else v)


# ---------------------------------------------------------------------------
# T3 — Music nodes
# ---------------------------------------------------------------------------

def build_music_nodes(
    charts_df: pd.DataFrame,
    songs_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
) -> tuple[Tensor, dict[str, int]]:
    """Build music node features and id-map.

    Returns:
        x_music: float32 Tensor of shape (N_m, 12)
        music_id_map: dict mapping song_id -> index (sorted, deterministic)

    Feature columns (12) — static metadata only. The full-series chart
    aggregates popularity, total_streams and dias_no_chart were REMOVED to
    avoid temporal leakage: they summarise the whole 2017-2021 window (incl.
    the test period) yet are reused unchanged at every snapshot by mask_until.
    See tests/test_node_feature_leakage.py.
        0-8: acoustic (acousticness, danceability, energy, instrumentalness,
             liveness, loudness, speechiness, valence, tempo) — z-scored
        9:   explicit — 0/1 (no z-score, binary)
        10:  song_type — encoded int, z-scored
        11:  acoustic_missing — 0/1 (no z-score, flag)
    """
    # Universe = top200 songs UNION viral50 songs that have acoustic features.
    # This matches the spec's ~6.469 estimate (actual: ~6.526) and avoids >48% imputation
    # when including all viral50 songs regardless of feature coverage.
    songs_with_features = set(songs_df["song_id"].dropna().tolist())
    all_chart_songs = set(charts_df["song_id"].dropna().tolist())
    top200_songs = set(charts_df[charts_df["chart"] == "top200"]["song_id"].dropna().tolist())
    viral50_songs = set(charts_df[charts_df["chart"] == "viral50"]["song_id"].dropna().tolist())
    # Include all songs that have features OR are in top200 (with imputation for any top200 gaps)
    universe = top200_songs | (viral50_songs & songs_with_features)
    universe_ids = sorted(universe)
    music_id_map: dict[str, int] = {sid: idx for idx, sid in enumerate(universe_ids)}
    N = len(universe_ids)

    # Songs dataframe indexed by song_id
    songs_indexed = songs_df.set_index("song_id")

    # Acoustic features: compute mean/std from songs that HAVE features (pre-imputation z-score)
    acoustic_raw = np.full((N, len(ACOUSTIC_COLS)), np.nan, dtype=np.float64)
    for col_i, col in enumerate(ACOUSTIC_COLS):
        if col in songs_indexed.columns:
            vals = songs_indexed[col].reindex(universe_ids)
            acoustic_raw[:, col_i] = vals.values

    # Compute z-score stats from non-missing rows only
    acoustic_means = np.nanmean(acoustic_raw, axis=0)
    acoustic_stds = np.nanstd(acoustic_raw, axis=0)
    acoustic_stds = np.where(acoustic_stds == 0, 1.0, acoustic_stds)

    # Identify missing rows
    acoustic_missing = np.any(np.isnan(acoustic_raw), axis=1).astype(np.float32)

    # Z-score (NaN rows get 0.0 = z-scored median)
    acoustic_z = np.where(
        np.isnan(acoustic_raw),
        0.0,
        (acoustic_raw - acoustic_means) / acoustic_stds,
    ).astype(np.float32)

    # Explicit (binary)
    if "explicit" in songs_indexed.columns:
        exp_raw = songs_indexed["explicit"].reindex(universe_ids)
        explicit = (
            exp_raw.map({"True": 1, "False": 0, True: 1, False: 0})
            .fillna(0)
            .values.astype(np.float32)
        )
    else:
        explicit = np.zeros(N, dtype=np.float32)

    # Song type (categorical → int → z-score)
    if "song_type" in songs_indexed.columns:
        st_raw = songs_indexed["song_type"].reindex(universe_ids)
        st_int = st_raw.map(_SONG_TYPE_MAP).fillna(0).values.astype(np.float64)
    else:
        st_int = np.zeros(N, dtype=np.float64)
    st_z = _zscore(st_int.reshape(-1, 1)).flatten().astype(np.float32)

    # Assemble (N, 12) — static metadata only; no full-series chart aggregates
    x = np.stack([
        *[acoustic_z[:, i] for i in range(len(ACOUSTIC_COLS))],  # 0-8
        explicit,          # 9
        st_z,              # 10
        acoustic_missing,  # 11
    ], axis=1)

    assert not np.isnan(x).any(), "NaN found in music node features after imputation"

    logger.info(
        "Music nodes: N=%d, acoustic_missing=%.1f%%",
        N, 100 * acoustic_missing.mean(),
    )

    return torch.tensor(x, dtype=torch.float32), music_id_map


# ---------------------------------------------------------------------------
# T4 — Artist nodes
# ---------------------------------------------------------------------------

def build_artist_nodes(
    artists_df: pd.DataFrame,
    music_id_map: dict[str, int],
    charts_df: pd.DataFrame,
    songs_df: pd.DataFrame,
) -> tuple[Tensor, dict[str, int]]:
    """Build artist node features and id-map.

    Returns:
        x_artist: float32 Tensor (N_a, 1)
        artist_id_map: dict artist_id -> index (sorted, deterministic)

    Feature columns (1) — static metadata only. The full-series chart
    aggregates num_hits, num_collab_hits and anos_no_chart were REMOVED to
    avoid temporal leakage: they count chart activity over the whole 2017-2021
    window (incl. the test period) and propagate to music via `performs`.
    See tests/test_node_feature_leakage.py.
        0: n_genres = len(genres_list) (z-scored)
    """
    # Find all artist_ids appearing in songs within music_id_map
    universe_songs = set(music_id_map.keys())
    songs_in_universe = songs_df[songs_df["song_id"].isin(universe_songs)].copy()

    # Parse artist_id column (stored as string-encoded list)
    if "artist_id" not in songs_in_universe.columns:
        raise ValueError("songs_df missing 'artist_id' column")

    songs_in_universe["artist_id_list"] = _parse_list_col(songs_in_universe["artist_id"])
    all_artist_ids: set[str] = set()
    for lst in songs_in_universe["artist_id_list"]:
        if isinstance(lst, list):
            all_artist_ids.update(lst)

    # Filter artists_df to those in all_artist_ids
    if "artist_id" not in artists_df.columns:
        raise ValueError("artists_df missing 'artist_id' column")

    filtered = artists_df[artists_df["artist_id"].isin(all_artist_ids)].copy()

    missing_ids = all_artist_ids - set(filtered["artist_id"].tolist())
    if missing_ids:
        logger.warning(
            "%d artist(s) referenced in songs but absent from artists_df "
            "(skipped — expected for non-BR artists from viral50-only songs): e.g. %s",
            len(missing_ids),
            list(missing_ids)[:3],
        )
        # Do NOT create zero-feature nodes: artist universe = br-artists-all_time.csv only

    # n_genres
    if "genres_list" not in filtered.columns:
        filtered["genres_list"] = filtered["genres"].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) else []
        )
    filtered["n_genres"] = filtered["genres_list"].apply(
        lambda v: len(v) if isinstance(v, list) else 0
    )

    # Deterministic ordering
    sorted_ids = sorted(filtered["artist_id"].tolist())
    artist_id_map: dict[str, int] = {aid: idx for idx, aid in enumerate(sorted_ids)}

    df = filtered.set_index("artist_id").reindex(sorted_ids)

    # Static metadata only: n_genres. Full-series chart counts (num_hits,
    # num_collab_hits, anos_no_chart) removed to avoid temporal leakage (P0).
    feat_raw = np.stack([
        df["n_genres"].fillna(0).values.astype(np.float64),
    ], axis=1)

    feat_z = _zscore(feat_raw).astype(np.float32)
    # After zscore, fill any remaining NaN (from constant cols) with 0
    feat_z = np.nan_to_num(feat_z, nan=0.0)

    logger.info("Artist nodes: N=%d", len(sorted_ids))

    return torch.tensor(feat_z, dtype=torch.float32), artist_id_map


# ---------------------------------------------------------------------------
# T5 — Genre nodes
# ---------------------------------------------------------------------------

def build_genre_nodes(
    artists_df: pd.DataFrame,
    artist_id_map: dict[str, int],
) -> tuple[Tensor, dict[str, int]]:
    """Build genre node embeddings (random init) and id-map.

    Returns:
        x_genre: float32 Tensor (N_g, 32) — random normal(0, 0.1), seed=0
        genre_id_map: dict genre_name -> index (sorted alphabetically)
    """
    # Genre universe = union of genres_list for artists in artist_id_map
    if "genres_list" not in artists_df.columns:
        artists_df = artists_df.copy()
        artists_df["genres_list"] = _parse_list_col(artists_df["genres"])

    relevant = artists_df[artists_df["artist_id"].isin(artist_id_map)]
    all_genres: set[str] = set()
    for lst in relevant["genres_list"]:
        if isinstance(lst, list):
            all_genres.update(lst)

    sorted_genres = sorted(all_genres)
    genre_id_map: dict[str, int] = {g: i for i, g in enumerate(sorted_genres)}
    N_g = len(sorted_genres)

    torch.manual_seed(0)
    x_genre = torch.empty(N_g, 32).normal_(0, 0.1)

    logger.info("Genre nodes: N=%d", N_g)

    return x_genre, genre_id_map
