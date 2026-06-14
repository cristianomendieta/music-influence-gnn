"""Preprocessing pipeline: rank score → MA-7d → min-max [0, 0.5] → floor 0.001.

Identical to Oliveira et al. BraSNAM 2025, Section 4.1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from music_diffusion_gnn.data.loaders import load_charts, load_release_dates
from music_diffusion_gnn.data.subset import build_subset, load_subset

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"

_MAX_RANK = {"top200": 200, "viral50": 50}
GLOBAL_START = pd.Timestamp("2017-01-01")


def _rank_to_score(rank: pd.Series, max_rank: int) -> pd.Series:
    return max_rank - rank + 1


def _normalize_series(s: pd.Series, window: int = 7, target_max: float = 0.5, floor: float = 0.001) -> pd.Series:
    """MA-7d → min-max [0, target_max] → floor."""
    smoothed = s.rolling(window=window, min_periods=1).mean()
    rng = smoothed.max() - smoothed.min()
    if rng > 0:
        normalized = (smoothed - smoothed.min()) / rng * target_max
    else:
        normalized = smoothed * 0.0
    return normalized.where(normalized > 0, floor)


def build_timeseries(
    raw_charts_path: Path | None = None,
    out_path: Path | None = None,
    *,
    subset_ids: list[str] | None = None,
    floor: float = 0.001,
    window: int = 7,
    target_max: float = 0.5,
) -> pd.DataFrame:
    """Build long-format timeseries parquet for the viral∩hit subset.

    Schema: song_id | chart | date | rank_score | y
    """
    if out_path is None:
        out_path = PROCESSED / "timeseries.parquet"

    if subset_ids is None:
        subset_path = PROCESSED / "subset_ids.json"
        if subset_path.exists():
            subset_ids = load_subset(subset_path)
        else:
            subset_ids = build_subset()

    charts = load_charts(raw_charts_path)
    subset_set = set(subset_ids)
    end_date = charts["date"].max()

    # Per-song start: max(release_date, 2017-01-01). NaT → GLOBAL_START.
    release_dates = load_release_dates()
    start_dates = release_dates.clip(lower=GLOBAL_START)

    records: list[pd.DataFrame] = []

    for chart_name, max_rank in _MAX_RANK.items():
        df_chart = charts[
            (charts["chart"] == chart_name) & (charts["song_id"].isin(subset_set))
        ].copy()
        df_chart["rank_score"] = _rank_to_score(df_chart["rank"], max_rank)
        # Dedup any (song_id, date) collisions by keeping best (max) rank_score.
        df_chart = (
            df_chart.groupby(["song_id", "date"], as_index=False)["rank_score"].max()
        )

        chart_records: list[pd.DataFrame] = []
        for song_id, grp in df_chart.groupby("song_id", sort=False):
            start = start_dates.get(song_id, pd.NaT)
            if pd.isna(start):
                start = GLOBAL_START
            start = max(start, GLOBAL_START)
            # A song released after the end of the chart window has no data here.
            if start > end_date:
                continue
            date_range = pd.date_range(start, end_date, freq="D")
            series = (
                grp.set_index("date")["rank_score"]
                .reindex(date_range, fill_value=0)
            )
            y = _normalize_series(series, window=window, target_max=target_max, floor=floor)
            chart_records.append(
                pd.DataFrame(
                    {
                        "song_id": song_id,
                        "chart": chart_name,
                        "date": date_range,
                        "rank_score": series.to_numpy(),
                        "y": y.to_numpy(),
                    }
                )
            )

        if chart_records:
            records.append(pd.concat(chart_records, ignore_index=True))

    result = pd.concat(records, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["song_id", "chart", "date"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    return result


def load_timeseries(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = PROCESSED / "timeseries.parquet"
    return pd.read_parquet(path)
