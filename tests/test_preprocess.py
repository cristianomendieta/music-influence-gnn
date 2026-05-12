"""Unit tests for data.preprocess — synthetic series, no real data."""
import numpy as np
import pandas as pd
import pytest

from music_diffusion_gnn.data.preprocess import _normalize_series, _rank_to_score


def test_rank_to_score_top200():
    ranks = pd.Series([1, 100, 200])
    scores = _rank_to_score(ranks, 200)
    assert list(scores) == [200, 101, 1]


def test_rank_to_score_viral50():
    ranks = pd.Series([1, 25, 50])
    scores = _rank_to_score(ranks, 50)
    assert list(scores) == [50, 26, 1]


def test_ma7d_smoothing():
    """A spike on day 3 should be spread over days 0-6 by the rolling window."""
    s = pd.Series([0.0] * 3 + [70.0] + [0.0] * 3)
    result = _normalize_series(s, window=7, target_max=0.5, floor=0.001)
    # The day of the spike should have a higher normalized value than neighbors
    assert result.iloc[3] > result.iloc[0]
    # Normalization output must be in (0, 0.5]
    assert result.max() <= 0.5 + 1e-9


def test_minmax_peak_is_0_5():
    """Series with one peak should normalize that peak to exactly 0.5."""
    s = pd.Series([0.0] * 5 + [100.0] * 5 + [0.0] * 5)
    result = _normalize_series(s, window=1, target_max=0.5, floor=0.001)
    assert abs(result.max() - 0.5) < 1e-9


def test_floor_replaces_zeros():
    """Days with rank_score == 0 (absent from chart) should become floor after normalization."""
    # Only one non-zero day so smoothed zeros exist
    s = pd.Series([0.0] * 10 + [50.0] + [0.0] * 10)
    result = _normalize_series(s, window=1, target_max=0.5, floor=0.001)
    # Exact-zero days → floor; the non-zero day → something > 0
    zero_days = result[s == 0.0]
    assert (zero_days == 0.001).all(), "Absent days must be replaced by floor=0.001"


def test_constant_zero_series():
    """A series of all zeros (song never in chart) should return all floor."""
    s = pd.Series([0.0] * 20)
    result = _normalize_series(s, window=7, target_max=0.5, floor=0.001)
    # When max == 0 we return zeros → then floor replaces them
    assert (result == 0.001).all()


def test_output_length_preserved():
    s = pd.Series(np.random.rand(100))
    result = _normalize_series(s)
    assert len(result) == 100
