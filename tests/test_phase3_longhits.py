"""Unit tests for days_on_chart / long_hit_mask (synthetic data, no real timeseries)."""
from __future__ import annotations

import pandas as pd

from music_diffusion_gnn.evaluation.longhits import days_on_chart, long_hit_mask


def _make_group(song_id: str, chart: str, n_on: int, m_off: int) -> pd.DataFrame:
    """n_on rows with rank_score > 0 followed by m_off rows with rank_score == 0."""
    rows = []
    for i in range(n_on):
        rows.append({"song_id": song_id, "chart": chart, "date": i, "rank_score": 1.0, "y": 0.05})
    for i in range(n_on, n_on + m_off):
        rows.append({"song_id": song_id, "chart": chart, "date": i, "rank_score": 0.0, "y": 0.001})
    return pd.DataFrame(rows)


def test_days_on_chart_ignores_dense_floor_days():
    df = _make_group("songA", "top200", n_on=50, m_off=200)
    counts = days_on_chart(df)
    assert counts[("songA", "top200")] == 50


def test_long_hit_mask_boundary_excludes_90_includes_91():
    df_below = _make_group("songBelow", "top200", n_on=90, m_off=10)
    df_above = _make_group("songAbove", "top200", n_on=91, m_off=10)
    ts_df = pd.concat([df_below, df_above], ignore_index=True)

    mask = long_hit_mask(ts_df, threshold_days=90)

    assert ("songAbove", "top200") in mask
    assert ("songBelow", "top200") not in mask


def test_long_hit_mask_multiple_groups_and_default_threshold():
    df_short = _make_group("songShort", "viral50", n_on=5, m_off=300)
    df_long = _make_group("songLong", "viral50", n_on=120, m_off=50)
    ts_df = pd.concat([df_short, df_long], ignore_index=True)

    mask = long_hit_mask(ts_df)

    assert mask == {("songLong", "viral50")}


def test_days_on_chart_index_levels():
    df = _make_group("songX", "top200", n_on=10, m_off=5)
    counts = days_on_chart(df)
    assert counts.index.names == ["song_id", "chart"]
