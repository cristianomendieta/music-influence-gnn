"""Unit tests for parametrized split regimes (dataset.SPLIT_REGIMES / temporal_split)."""
from __future__ import annotations

import pandas as pd
import pytest

from music_diffusion_gnn.training.dataset import (
    SPLIT_REGIMES,
    TEST_START_WEEK,
    TRAIN_END_WEEK,
    get_split_regime,
    temporal_split,
)


def _weekly_df(weeks: range) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "song_id": ["s1"] * len(weeks),
            "chart": ["top200"] * len(weeks),
            "week": list(weeks),
            "y_week": [0.1] * len(weeks),
        }
    )


def test_current_regime_matches_backward_compat_constants():
    r = get_split_regime("current")
    assert r.train_end_week == TRAIN_END_WEEK
    assert r.test_start_week == TEST_START_WEEK
    assert r.test_end_week is None


def test_temporal_split_default_regime_reproduces_today():
    df = _weekly_df(range(0, 261))
    default = temporal_split(df)
    explicit = temporal_split(df, regime="current")
    for key in ("train", "val", "test"):
        assert default[key]["week"].tolist() == explicit[key]["week"].tolist()
    assert len(default["train"]) + len(default["val"]) + len(default["test"]) == len(df)


def test_temporal_split_pre_pandemia_regime_is_bounded_and_excludes_2020_plus():
    df = _weekly_df(range(0, 261))
    splits = temporal_split(df, regime="pre_pandemia")
    r = SPLIT_REGIMES["pre_pandemia"]
    assert splits["train"]["week"].max() == r.train_end_week
    assert splits["test"]["week"].max() == r.test_end_week
    # Weeks after the bounded test window belong to no split (dropped, not leaked)
    covered = set(splits["train"]["week"]) | set(splits["val"]["week"]) | set(splits["test"]["week"])
    assert max(covered) == r.test_end_week
    assert len(covered) < len(df)


def test_temporal_split_splits_are_pairwise_disjoint_for_both_regimes():
    df = _weekly_df(range(0, 261))
    for regime in SPLIT_REGIMES:
        splits = temporal_split(df, regime=regime)
        train_wk = set(splits["train"]["week"])
        val_wk = set(splits["val"]["week"])
        test_wk = set(splits["test"]["week"])
        assert train_wk.isdisjoint(val_wk)
        assert val_wk.isdisjoint(test_wk)
        assert train_wk.isdisjoint(test_wk)


def test_get_split_regime_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_split_regime("nonexistent_regime")


def test_get_split_regime_is_idempotent_on_a_regime_object():
    r = SPLIT_REGIMES["current"]
    assert get_split_regime(r) is r
