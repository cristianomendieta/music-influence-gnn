"""Long-hit subgroup: songs with many genuine days on chart."""
from __future__ import annotations

import pandas as pd


def days_on_chart(ts_df: pd.DataFrame) -> pd.Series:
    """Genuine days on chart per (song_id, chart), from the daily DataFrame.

    Must use rank_score > 0, NOT the dense calendar span: timeseries.parquet
    is 100% dense (one row per day for the whole span), with off-chart days
    filled as rank_score == 0 / y floored at 0.001. Counting span or y > floor
    would call ~97% of songs "long hits".
    """
    on_chart = ts_df["rank_score"] > 0
    return ts_df.loc[on_chart].groupby(["song_id", "chart"]).size()


def long_hit_mask(ts_df: pd.DataFrame, *, threshold_days: int = 90) -> set[tuple[str, str]]:
    """Set of (song_id, chart) with days_on_chart strictly above threshold_days."""
    counts = days_on_chart(ts_df)
    return set(counts[counts > threshold_days].index)
