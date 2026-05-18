"""Edge builders for all 4 edge types in the heterogeneous graph."""
from __future__ import annotations

import ast
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from music_diffusion_gnn.graph.temporal import week_index

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "data"

EdgeData = dict[str, Tensor]

# ---------------------------------------------------------------------------
# T6 — (artist, performs, music)
# ---------------------------------------------------------------------------

def build_performs(
    charts_df: pd.DataFrame,
    songs_df: pd.DataFrame,
    music_id_map: dict[str, int],
    artist_id_map: dict[str, int],
) -> EdgeData:
    """Build (artist, performs, music) directed edges.

    edge_attr columns: [role (0=main,1=feat), position_in_list, first_seen_week]
    first_seen_week = week_index(min date of song across all charts).
    """
    # first_seen_week per song
    charts_df = charts_df.copy()
    charts_df["date"] = pd.to_datetime(charts_df["date"])
    first_week: dict[str, int] = {}
    for sid, grp in charts_df.groupby("song_id"):
        if sid in music_id_map:
            try:
                first_week[sid] = week_index(grp["date"].min().date())
            except ValueError:
                first_week[sid] = 0

    # Parse artist_id from songs_df
    songs_sub = songs_df[songs_df["song_id"].isin(music_id_map)].copy()
    songs_sub["artist_id_list"] = songs_sub["artist_id"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    )

    rows_src, rows_dst, rows_attr = [], [], []
    skipped_artists, skipped_songs = 0, 0

    for _, row in songs_sub.iterrows():
        sid = row["song_id"]
        if sid not in music_id_map:
            skipped_songs += 1
            continue
        music_idx = music_id_map[sid]
        fsw = first_week.get(sid, 0)

        artist_list = row["artist_id_list"]
        if not isinstance(artist_list, list):
            continue

        for pos, aid in enumerate(artist_list):
            if aid not in artist_id_map:
                skipped_artists += 1
                continue
            artist_idx = artist_id_map[aid]
            role = 0 if pos == 0 else 1
            rows_src.append(artist_idx)
            rows_dst.append(music_idx)
            rows_attr.append([role, pos, fsw])

    if skipped_artists > 0:
        logger.warning("build_performs: skipped %d artist references not in artist_id_map", skipped_artists)

    if not rows_src:
        return {
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_attr": torch.zeros((0, 3), dtype=torch.float32),
        }

    edge_index = torch.tensor([rows_src, rows_dst], dtype=torch.long)
    edge_attr = torch.tensor(rows_attr, dtype=torch.float32)

    logger.info("performs edges: %d", edge_index.shape[1])
    return {"edge_index": edge_index, "edge_attr": edge_attr}


# ---------------------------------------------------------------------------
# T7 — (artist, has_genre, genre)
# ---------------------------------------------------------------------------

def build_has_genre(
    artists_df: pd.DataFrame,
    artist_id_map: dict[str, int],
    genre_id_map: dict[str, int],
) -> EdgeData:
    """Build (artist, has_genre, genre) directed edges.

    No edge features. first_seen_week = 0 for all (no temporal info on genre affiliation).
    """
    if "genres_list" not in artists_df.columns:
        artists_df = artists_df.copy()
        artists_df["genres_list"] = artists_df["genres"].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) else []
        )

    src, dst = [], []
    for _, row in artists_df.iterrows():
        aid = row["artist_id"]
        if aid not in artist_id_map:
            continue
        a_idx = artist_id_map[aid]
        for genre in row["genres_list"]:
            if genre not in genre_id_map:
                continue
            src.append(a_idx)
            dst.append(genre_id_map[genre])

    if not src:
        return {
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "first_seen_week": torch.zeros(0, dtype=torch.long),
        }

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    first_seen_week = torch.zeros(len(src), dtype=torch.long)

    logger.info("has_genre edges: %d", len(src))
    return {"edge_index": edge_index, "first_seen_week": first_seen_week}


# ---------------------------------------------------------------------------
# T8 — (music, cotrajectory, music)
# ---------------------------------------------------------------------------

def build_cotrajectory(
    charts_df: pd.DataFrame,
    music_id_map: dict[str, int],
) -> EdgeData:
    """Build (music, cotrajectory, music) directed edges.

    2 parallel edges per pair if they co-occur >= 7 days in BOTH charts.
    Direction: i→j if i entered chart before j; tie-break = lexicographic song_id.

    edge_attr columns: [days_together, avg_position_distance, chart (0=viral50,1=top200),
                        first_seen_week]
    """
    t_start = time.time()

    charts_df = charts_df.copy()
    charts_df["date"] = pd.to_datetime(charts_df["date"])

    chart_map = {"viral50": 0, "top200": 1}
    all_edges: list[tuple[int, int, int, float, int, int]] = []

    for chart_label, chart_int in [("viral50", 0), ("top200", 1)]:
        sub = charts_df[
            (charts_df["chart"] == chart_label) &
            (charts_df["song_id"].isin(music_id_map))
        ].copy()

        if sub.empty:
            continue

        # first appearance date per song in this chart (for direction)
        first_date: dict[str, pd.Timestamp] = (
            sub.groupby("song_id")["date"].min().to_dict()
        )

        # Sort dates for deterministic processing and first_seen_week computation
        daily_groups = sorted(sub.groupby("date"), key=lambda x: x[0])

        # pair accumulators: (si, sj) with si < sj lexicographically
        pair_count: dict[tuple[str, str], int] = defaultdict(int)
        pair_rank_sum: dict[tuple[str, str], float] = defaultdict(float)
        pair_dates7: dict[tuple[str, str], list] = defaultdict(list)

        for dt, grp in daily_groups:
            in_map = grp[grp["song_id"].isin(music_id_map)]
            songs_list = sorted(in_map["song_id"].tolist())
            ranks = dict(zip(in_map["song_id"], in_map["rank"].fillna(0)))

            n = len(songs_list)
            for k in range(n):
                for l in range(k + 1, n):
                    si, sj = songs_list[k], songs_list[l]
                    # Canonical ordering: smaller string first
                    key = (si, sj)
                    pair_count[key] += 1
                    pair_rank_sum[key] += abs(ranks[si] - ranks[sj])
                    if len(pair_dates7[key]) < 7:
                        pair_dates7[key].append(dt)

        # Create edges for pairs with >= 7 co-occurrence days
        for (si, sj), count in pair_count.items():
            if count < 7:
                continue

            avg_pos_dist = pair_rank_sum[(si, sj)] / count
            dates7 = pair_dates7[(si, sj)]
            # first_seen_week = week the pair became eligible (7th accumulated day)
            fsw_date = dates7[6] if len(dates7) >= 7 else dates7[-1]
            try:
                fsw = week_index(fsw_date.date())
            except ValueError:
                fsw = 0

            # Determine direction
            d_si = first_date.get(si)
            d_sj = first_date.get(sj)
            if d_si is not None and d_sj is not None and d_si < d_sj:
                src_id, dst_id = si, sj
            elif d_si is not None and d_sj is not None and d_si > d_sj:
                src_id, dst_id = sj, si
            else:
                # Tie: lexicographic
                src_id, dst_id = (si, sj) if si <= sj else (sj, si)

            src_idx = music_id_map[src_id]
            dst_idx = music_id_map[dst_id]
            all_edges.append((src_idx, dst_idx, count, avg_pos_dist, chart_int, fsw))

    elapsed = time.time() - t_start
    logger.info(
        "build_cotrajectory: %d edges in %.1fs",
        len(all_edges), elapsed,
    )
    if elapsed > 300:
        logger.warning(
            "build_cotrajectory took %.1fs (>5min). "
            "Consider aggregating by week instead of day.",
            elapsed,
        )

    if not all_edges:
        return {
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_attr": torch.zeros((0, 4), dtype=torch.float32),
        }

    arr = np.array(all_edges, dtype=np.float64)
    edge_index = torch.tensor(arr[:, :2].T.astype(np.int64), dtype=torch.long)
    # edge_attr: [days_together, avg_pos_dist, chart, first_seen_week]
    edge_attr = torch.tensor(arr[:, 2:].astype(np.float32), dtype=torch.float32)

    return {"edge_index": edge_index, "edge_attr": edge_attr}


# ---------------------------------------------------------------------------
# T9 — (genre, cooccurs, genre)
# ---------------------------------------------------------------------------

def build_cooccurs(
    genre_id_map: dict[str, int],
    year_range: tuple[int, int] = (2017, 2021),
    data_dir: Path | None = None,
) -> EdgeData:
    """Build (genre, cooccurs, genre) edges (symmetric — both directions included).

    Iterates MGD+ genre_network CSVs year by year.
    first_seen_week = (first_year - 2017) * 52 (proxy: week-1 of the year).
    Attributes updated with snapshot of the most recent year for each pair.

    edge_attr columns: [weight, avg_popularity, avg_streams, first_seen_week]
    """
    if data_dir is None:
        data_dir = _DATA / "genre_network"

    # pair -> {first_seen_week, weight, avg_pop, avg_streams}
    pair_attrs: dict[tuple[int, int], dict] = {}

    for year in range(year_range[0], year_range[1] + 1):
        fpath = data_dir / f"br-genre_network-{year}.csv"
        if not fpath.exists():
            logger.warning("Genre network file missing: %s", fpath)
            continue

        # Genre network CSVs use comma separator
        df = pd.read_csv(fpath)
        fsw_year = (year - 2017) * 52

        for row in df.itertuples(index=False):
            src_name = row.Source
            dst_name = row.Target
            if src_name not in genre_id_map or dst_name not in genre_id_map:
                continue

            src_idx = genre_id_map[src_name]
            dst_idx = genre_id_map[dst_name]

            weight = float(getattr(row, "Weight", 1.0))
            avg_pop = float(getattr(row, "Avg_Popularity", 0.0))
            avg_streams = float(getattr(row, "Avg_Streams", 0.0))

            # Both directions (undirected)
            for key in [(src_idx, dst_idx), (dst_idx, src_idx)]:
                if key not in pair_attrs:
                    pair_attrs[key] = {
                        "first_seen_week": fsw_year,
                        "weight": weight,
                        "avg_pop": avg_pop,
                        "avg_streams": avg_streams,
                    }
                else:
                    # Update to latest snapshot
                    pair_attrs[key]["weight"] = weight
                    pair_attrs[key]["avg_pop"] = avg_pop
                    pair_attrs[key]["avg_streams"] = avg_streams

    if not pair_attrs:
        return {
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_attr": torch.zeros((0, 4), dtype=torch.float32),
        }

    # Deterministic ordering
    sorted_edges = sorted(pair_attrs.items())
    srcs = [k[0] for k, _ in sorted_edges]
    dsts = [k[1] for k, _ in sorted_edges]
    attrs = [
        [v["weight"], v["avg_pop"], v["avg_streams"], v["first_seen_week"]]
        for _, v in sorted_edges
    ]

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)

    logger.info("cooccurs edges: %d", len(srcs))
    return {"edge_index": edge_index, "edge_attr": edge_attr}
