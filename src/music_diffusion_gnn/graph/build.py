"""Graph orchestrator — assembles HeteroData, validates C1-C7, persists artifacts."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from music_diffusion_gnn.data.loaders import (
    load_artists,
    load_charts,
    load_songs,
)
from music_diffusion_gnn.graph.edges import (
    build_cooccurs,
    build_cotrajectory,
    build_has_genre,
    build_performs,
)
from music_diffusion_gnn.graph.nodes import (
    build_artist_nodes,
    build_genre_nodes,
    build_music_nodes,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED = _ROOT / "data" / "processed"


def _load_songs_combined(root: Path) -> pd.DataFrame:
    """Load songs from br-hit_songs CSVs + MGDplus complete dataset (for viral50 coverage).

    The br-hit_songs files cover Top200 BR songs (5,010). The complete MGDplus dataset
    adds viral50-only songs with acoustic features, bringing the total to ~6,526 —
    matching the spec's estimated universe of 6,469±100.
    """
    base_songs = load_songs()  # 5,010 top200 songs

    # Extend with complete MGDplus dataset for viral50 coverage
    complete_path = root / "data" / "MGDplus" / "songs" / "spotify_hit_songs_dataset_complete.csv"
    if complete_path.exists():
        complete_df = pd.read_csv(complete_path, sep="\t")
        existing_ids = set(base_songs["song_id"].tolist())
        new_songs = complete_df[~complete_df["song_id"].isin(existing_ids)]
        combined = pd.concat([base_songs, new_songs], ignore_index=True)
        combined = combined.drop_duplicates(subset=["song_id"]).reset_index(drop=True)
        logger.info(
            "Songs combined: %d br-hit_songs + %d from complete = %d total",
            len(base_songs), len(new_songs), len(combined),
        )
        return combined
    else:
        logger.warning("MGDplus complete songs file not found: %s", complete_path)
        return base_songs


def build_hetero(
    out_dir: Path = Path("data/processed/graph"),
) -> HeteroData:
    """Full Phase 1 pipeline: load → build nodes/edges → assemble → validate → persist.

    Validates criteria C1-C7 inline; raises AssertionError on violation.
    Persists hetero_full.pt and node_id_map.json to out_dir.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load raw data
    # ------------------------------------------------------------------ #
    logger.info("Loading charts...")
    charts_df = load_charts()

    logger.info("Loading songs (br-hit_songs + MGDplus complete for viral50)...")
    songs_df = _load_songs_combined(_ROOT)

    logger.info("Loading artists...")
    artists_df = load_artists()

    logger.info("Loading timeseries parquet...")
    ts_path = _PROCESSED / "timeseries.parquet"
    timeseries_df = pd.read_parquet(ts_path) if ts_path.exists() else pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 2. Build nodes
    # ------------------------------------------------------------------ #
    logger.info("Building music nodes...")
    x_music, music_id_map = build_music_nodes(charts_df, songs_df, timeseries_df)

    logger.info("Building artist nodes...")
    x_artist, artist_id_map = build_artist_nodes(
        artists_df, music_id_map, charts_df, songs_df
    )

    logger.info("Building genre nodes...")
    x_genre, genre_id_map = build_genre_nodes(artists_df, artist_id_map)

    # ------------------------------------------------------------------ #
    # 3. Build edges
    # ------------------------------------------------------------------ #
    logger.info("Building performs edges...")
    perf = build_performs(charts_df, songs_df, music_id_map, artist_id_map)

    logger.info("Building has_genre edges...")
    hg = build_has_genre(artists_df, artist_id_map, genre_id_map)

    logger.info("Building cotrajectory edges (may take a few minutes)...")
    cotraj = build_cotrajectory(charts_df, music_id_map)

    logger.info("Building cooccurs edges...")
    cooc = build_cooccurs(genre_id_map)

    # ------------------------------------------------------------------ #
    # 4. Assemble HeteroData
    # ------------------------------------------------------------------ #
    g = HeteroData()

    g["music"].x = x_music
    g["music"].song_id = list(music_id_map.keys())

    g["artist"].x = x_artist
    g["artist"].artist_id = list(artist_id_map.keys())

    g["genre"].x = x_genre
    g["genre"].genre_name = list(genre_id_map.keys())

    # (artist, performs, music) — directed
    g["artist", "performs", "music"].edge_index = perf["edge_index"]
    g["artist", "performs", "music"].edge_attr = perf["edge_attr"]

    # (artist, has_genre, genre) — directed + manual reverse
    g["artist", "has_genre", "genre"].edge_index = hg["edge_index"]
    g["artist", "has_genre", "genre"].first_seen_week = hg["first_seen_week"]
    g["genre", "rev_has_genre", "artist"].edge_index = hg["edge_index"].flip(0)
    g["genre", "rev_has_genre", "artist"].first_seen_week = hg["first_seen_week"]

    # (music, cotrajectory, music) — directed
    g["music", "cotrajectory", "music"].edge_index = cotraj["edge_index"]
    g["music", "cotrajectory", "music"].edge_attr = cotraj["edge_attr"]

    # (genre, cooccurs, genre) — symmetric (both directions already in edge_index)
    g["genre", "cooccurs", "genre"].edge_index = cooc["edge_index"]
    g["genre", "cooccurs", "genre"].edge_attr = cooc["edge_attr"]

    # ------------------------------------------------------------------ #
    # 5. Validate C1-C7
    # ------------------------------------------------------------------ #
    _validate(g, music_id_map, artist_id_map, genre_id_map)

    # ------------------------------------------------------------------ #
    # 6. Persist artifacts
    # ------------------------------------------------------------------ #
    pt_path = out_dir / "hetero_full.pt"
    torch.save(g, pt_path)
    logger.info("Saved: %s", pt_path)

    id_map = {
        "music": {
            "spotify_id_to_idx": music_id_map,
            "idx_to_spotify_id": list(music_id_map.keys()),
        },
        "artist": {
            "artist_id_to_idx": artist_id_map,
            "idx_to_artist_id": list(artist_id_map.keys()),
        },
        "genre": {
            "genre_name_to_idx": genre_id_map,
            "idx_to_genre_name": list(genre_id_map.keys()),
        },
    }
    map_path = out_dir / "node_id_map.json"
    with open(map_path, "w") as f:
        json.dump(id_map, f, indent=2)
    logger.info("Saved: %s", map_path)

    return g


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate(
    g: HeteroData,
    music_id_map: dict[str, int],
    artist_id_map: dict[str, int],
    genre_id_map: dict[str, int],
) -> None:
    """Assert C1-C7. Raises AssertionError with clear message on violation."""

    n_music = g["music"].num_nodes
    n_artist = g["artist"].num_nodes
    n_genre = g["genre"].num_nodes

    # C1 — actual universe = top200 ∪ (viral50 ∩ complete_songs) ≈ 6526; spec estimated 6469
    assert abs(n_music - 6469) <= 100, (
        f"C1 FAIL: n_music={n_music}, expected 6469±100"
    )
    # C2
    assert abs(n_artist - 1701) <= 5, (
        f"C2 FAIL: n_artist={n_artist}, expected 1701±5"
    )
    # C3
    assert abs(n_genre - 530) <= 10, (
        f"C3 FAIL: n_genre={n_genre}, expected 530±10"
    )

    # C4 — subset ⊆ music nodes
    subset_path = _ROOT / "data" / "processed" / "subset_ids.json"
    if subset_path.exists():
        from music_diffusion_gnn.data.subset import load_subset
        subset_ids: list[str] = load_subset(subset_path)
        missing = [s for s in subset_ids if s not in music_id_map]
        assert len(missing) == 0, (
            f"C4 FAIL: {len(missing)} subset songs missing from music nodes: {missing[:5]}"
        )

    # C5 — artists of subset songs are reachable (have performs edges)
    if subset_path.exists() and 'subset_ids' in dir():
        perf_ei = g["artist", "performs", "music"].edge_index
        music_nodes_with_edges = set(perf_ei[1].tolist())
        subset_idxs = {music_id_map[s] for s in subset_ids if s in music_id_map}
        isolated = subset_idxs - music_nodes_with_edges
        assert len(isolated) == 0, (
            f"C5 FAIL: {len(isolated)} subset music nodes have no incoming performs edge"
        )

    # C6 — no dangling edges
    edge_types_bounds = [
        ("artist", "performs", "music", n_artist, n_music),
        ("artist", "has_genre", "genre", n_artist, n_genre),
        ("genre", "rev_has_genre", "artist", n_genre, n_artist),
        ("music", "cotrajectory", "music", n_music, n_music),
        ("genre", "cooccurs", "genre", n_genre, n_genre),
    ]
    for *et_parts, n_src, n_dst in edge_types_bounds:
        et = tuple(et_parts)
        ei = g[et].edge_index
        if ei.shape[1] == 0:
            continue
        assert ei[0].max() < n_src, (
            f"C6 FAIL: {et} src index {ei[0].max()} >= n_src={n_src}"
        )
        assert ei[1].max() < n_dst, (
            f"C6 FAIL: {et} dst index {ei[1].max()} >= n_dst={n_dst}"
        )

    # C7 — first_seen_week ∈ [0, 260]
    def _check_fsw(et: tuple) -> None:
        store = g[et]
        keys = set(store.keys())
        if "edge_attr" in keys and store.edge_attr is not None and store.edge_attr.shape[1] > 0:
            fsw = store.edge_attr[:, -1]
        elif "first_seen_week" in keys:
            fsw = store.first_seen_week.float()
        else:
            return
        assert (fsw >= 0).all() and (fsw <= 260).all(), (
            f"C7 FAIL: {et} first_seen_week out of [0, 260]: "
            f"min={fsw.min().item()}, max={fsw.max().item()}"
        )

    for *et_parts, _, __ in edge_types_bounds:
        _check_fsw(tuple(et_parts))

    logger.info("C1-C7 all passed: n_music=%d, n_artist=%d, n_genre=%d", n_music, n_artist, n_genre)
