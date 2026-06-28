"""Unit tests for SIR Mode-1 weekly reconstruction (synthetic data, no real timeseries)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from music_diffusion_gnn.baselines.sir import SIRFit, fit_sir
from music_diffusion_gnn.evaluation.sir_eval import run_sir_mode1, sir_weekly_from_fit


def _make_daily_y(n: int = 200, start: str = "2018-01-01") -> pd.Series:
    """Synthetic daily series of plausible SIR shape, date-indexed ascending."""
    t = np.arange(n, dtype=float)
    # simple bump shape, clipped into [1e-6, 1-1e-6] like fit_sir's I0
    y = 0.01 + 0.04 * np.exp(-((t - n / 3) ** 2) / (2 * (n / 10) ** 2))
    dates = pd.date_range(start=start, periods=n, freq="D")
    return pd.Series(y, index=pd.DatetimeIndex(dates))


def test_sir_weekly_from_fit_values_in_range_and_week_count():
    daily_y = _make_daily_y(n=200)
    fit = SIRFit(beta=0.3, gamma=0.1, R0=3.0, rmse=0.0, converged=True, n_iter=1)

    weekly = sir_weekly_from_fit(daily_y, fit)

    assert isinstance(weekly, pd.Series)
    assert weekly.index.name == "week"
    assert weekly.index.is_monotonic_increasing
    expected_weeks = len(daily_y) // 7
    assert abs(len(weekly) - expected_weeks) <= 1
    assert weekly.between(0, 0.5).all(), f"out of range: min={weekly.min()}, max={weekly.max()}"


def test_sir_weekly_from_fit_uses_fit_sir_roundtrip():
    """Fit a real SIR curve with fit_sir, then check the weekly reconstruction is sane."""
    beta_true, gamma_true = 0.3, 0.1
    n = 150
    t = np.arange(n, dtype=float)
    I0 = 0.01
    from scipy.integrate import odeint

    from music_diffusion_gnn.baselines.sir import _sir_ode

    y0 = [1.0 - I0, I0, 0.0]
    sol = odeint(_sir_ode, y0, t, args=(beta_true, gamma_true))
    y = sol[:, 1]

    fit = fit_sir(y)
    assert fit.converged

    dates = pd.date_range(start="2019-03-01", periods=n, freq="D")
    daily_y = pd.Series(y, index=pd.DatetimeIndex(dates))

    weekly = sir_weekly_from_fit(daily_y, fit)
    assert weekly.between(0, 0.5).all()
    assert abs(len(weekly) - n // 7) <= 1


def test_sir_weekly_from_fit_accepts_pandas_series_row():
    """fit may be a row of sir_params_df (pandas Series with attribute-style access)."""
    daily_y = _make_daily_y(n=70)
    params = pd.Series({"beta": 0.3, "gamma": 0.1, "R0": 3.0, "rmse": 0.0, "converged": True, "n_iter": 1})

    weekly = sir_weekly_from_fit(daily_y, params)

    assert weekly.between(0, 0.5).all()
    assert len(weekly) > 0


def test_sir_weekly_from_fit_drops_out_of_range_weeks():
    """Dates before 2017 or after the graph's week-260 horizon are dropped, not raised."""
    daily_y = _make_daily_y(n=30, start="2016-12-20")  # straddles ISO-year boundary into 2017
    fit = SIRFit(beta=0.3, gamma=0.1, R0=3.0, rmse=0.0, converged=True, n_iter=1)

    weekly = sir_weekly_from_fit(daily_y, fit)  # must not raise

    assert (weekly.index >= 0).all()
    assert (weekly.index <= 260).all()


def _build_ts_df(song_chart_specs: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    """song_chart_specs: list of (song_id, chart, n_days, start_date)."""
    frames = []
    for song_id, chart, n, start in song_chart_specs:
        daily = _make_daily_y(n=n, start=start)
        frames.append(
            pd.DataFrame(
                {
                    "song_id": song_id,
                    "chart": chart,
                    "date": daily.index,
                    "y": daily.to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_run_sir_mode1_returns_expected_columns():
    ts_df = _build_ts_df(
        [
            ("songA", "top200", 100, "2018-01-01"),
            ("songB", "viral50", 80, "2018-06-01"),
        ]
    )
    sir_params_df = pd.DataFrame(
        {
            "beta": [0.3, 0.4],
            "gamma": [0.1, 0.15],
            "R0": [3.0, 2.6],
            "rmse": [0.0, 0.0],
            "converged": [True, True],
            "n_iter": [1, 1],
        },
        index=pd.MultiIndex.from_tuples(
            [("songA", "top200"), ("songB", "viral50")], names=["song_id", "chart"]
        ),
    )

    out = run_sir_mode1(ts_df, sir_params_df)

    assert list(out.columns) == ["song_id", "chart", "week", "y_pred_sir"]
    assert set(out["song_id"].unique()) == {"songA", "songB"}
    assert out["y_pred_sir"].between(0, 0.5).all()


def test_run_sir_mode1_inner_join_skips_missing_fit():
    """A song present in ts_df but absent from sir_params_df is skipped, not raised."""
    ts_df = _build_ts_df(
        [
            ("songA", "top200", 60, "2018-01-01"),
            ("songNoFit", "top200", 60, "2018-01-01"),
        ]
    )
    sir_params_df = pd.DataFrame(
        {"beta": [0.3], "gamma": [0.1], "R0": [3.0], "rmse": [0.0], "converged": [True], "n_iter": [1]},
        index=pd.MultiIndex.from_tuples([("songA", "top200")], names=["song_id", "chart"]),
    )

    out = run_sir_mode1(ts_df, sir_params_df)

    assert set(out["song_id"].unique()) == {"songA"}
