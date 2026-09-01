"""Long-hit subgroup and the on-chart recorte, both read off ``rank_score``."""
from __future__ import annotations

import pandas as pd

_MAX_WEEK = 260


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


def onchart_weeks(ts_df: pd.DataFrame, *, max_week: int = _MAX_WEEK) -> set[tuple[str, str, int]]:
    """Set of ``(song_id, chart, week)`` genuinely on chart (ADR-0004).

    A week counts when the song charted on at least one of its days
    (``rank_score > 0``). The complement is the *floor*: absence of an
    observation, not low popularity — which is why ~95% of Mode-2 targets sit
    there and why the on-chart reading is the principal one.
    """
    df = ts_df.loc[ts_df["rank_score"] > 0, ["song_id", "chart", "date"]].copy()
    iso = df["date"].dt.isocalendar()
    df["week"] = (iso["year"].astype(int) - 2017) * 52 + (iso["week"].astype(int) - 1)
    df = df[(df["week"] >= 0) & (df["week"] <= max_week)]
    return set(zip(df["song_id"], df["chart"], df["week"].astype(int)))
