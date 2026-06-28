"""Unit tests for persistence_multistep (synthetic data, no real timeseries)."""
from __future__ import annotations

import pandas as pd

from music_diffusion_gnn.models.baselines import persistence_multistep


def _make_weekly_df() -> pd.DataFrame:
    rows = [
        {"song_id": "songA", "chart": "top200", "week": 10, "y_week": 0.2},
        {"song_id": "songA", "chart": "top200", "week": 11, "y_week": 0.3},
        {"song_id": "songA", "chart": "top200", "week": 12, "y_week": 0.4},
        {"song_id": "songB", "chart": "viral50", "week": 20, "y_week": 0.05},
    ]
    return pd.DataFrame(rows)


def test_persistence_multistep_frozen_across_horizons():
    weekly_df = _make_weekly_df()
    origins = pd.DataFrame([{"song_id": "songA", "chart": "top200", "week": 10}])
    ks = (1, 2, 4)

    result = persistence_multistep(weekly_df, origins, ks)

    assert len(result) == len(origins) * len(ks)
    assert list(result.columns) == ["song_id", "chart", "origin_week", "k", "y_pred"]

    sub = result[(result["song_id"] == "songA") & (result["chart"] == "top200")]
    assert sub["origin_week"].eq(10).all()
    assert set(sub["k"]) == set(ks)
    assert sub["y_pred"].nunique() == 1
    assert sub["y_pred"].iloc[0] == 0.2


def test_persistence_multistep_never_reads_future_weeks():
    weekly_df = _make_weekly_df()
    origins = pd.DataFrame([{"song_id": "songA", "chart": "top200", "week": 10}])

    result = persistence_multistep(weekly_df, origins, ks=(1, 2, 4))

    # y_week at week 10 is 0.2; weeks 11 (0.3) and 12 (0.4) must never leak in.
    assert (result["y_pred"] == 0.2).all()
    assert not (result["y_pred"] == 0.3).any()
    assert not (result["y_pred"] == 0.4).any()


def test_persistence_multistep_multiple_origins():
    weekly_df = _make_weekly_df()
    origins = pd.DataFrame(
        [
            {"song_id": "songA", "chart": "top200", "week": 11},
            {"song_id": "songB", "chart": "viral50", "week": 20},
        ]
    )
    ks = (1, 2)

    result = persistence_multistep(weekly_df, origins, ks)

    assert len(result) == len(origins) * len(ks)

    songa_rows = result[result["song_id"] == "songA"]
    assert songa_rows["y_pred"].eq(0.3).all()

    songb_rows = result[result["song_id"] == "songB"]
    assert songb_rows["y_pred"].eq(0.05).all()


def test_persistence_multistep_missing_origin_week_falls_back_to_zero():
    weekly_df = _make_weekly_df()
    origins = pd.DataFrame([{"song_id": "songA", "chart": "top200", "week": 999}])

    result = persistence_multistep(weekly_df, origins, ks=(1, 2, 4))

    assert (result["y_pred"] == 0.0).all()
