"""Unit tests for SIR Mode-1 weekly reconstruction (synthetic data, no real timeseries)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from music_diffusion_gnn.baselines.sir import SIRFit, fit_sir
from music_diffusion_gnn.evaluation.sir_eval import (
    run_sir_mode1,
    run_sir_mode2,
    sir_causal_forecast,
    sir_weekly_from_fit,
)
from music_diffusion_gnn.training.dataset import TEST_START_WEEK


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


# --- T6: sir_causal_forecast + run_sir_mode2 -------------------------------------


def _make_bump_y(n: int, peak_frac: float = 1.0 / 3) -> np.ndarray:
    """Synthetic daily series with a clear bump/peak, values in [1e-6, 0.5]."""
    t = np.arange(n, dtype=float)
    return 0.01 + 0.04 * np.exp(-((t - n * peak_frac) ** 2) / (2 * (n / 10) ** 2))


def test_sir_causal_forecast_converges_and_produces_all_horizons():
    daily_y_upto_w = _make_bump_y(n=120)

    out = sir_causal_forecast(daily_y_upto_w, (1, 2, 4), min_hist_weeks=4)

    assert out is not None
    assert set(out.keys()) == {1, 2, 4}
    for k, v in out.items():
        assert np.isfinite(v)
        assert 0.0 <= v <= 0.5 + 1e-6


def test_sir_causal_forecast_short_history_returns_none():
    # < 4 weeks (28 days) of on-chart (y > 0) days -> None, regardless of convergence.
    daily_y_upto_w = _make_bump_y(n=15)

    out = sir_causal_forecast(daily_y_upto_w, (1, 2, 4), min_hist_weeks=4)

    assert out is None


def test_sir_causal_forecast_nonconvergent_returns_none(monkeypatch):
    """If fit_sir reports converged=False, sir_causal_forecast must return None
    even with plenty of history."""
    import music_diffusion_gnn.evaluation.sir_eval as sir_eval_mod

    daily_y_upto_w = _make_bump_y(n=120)

    def _fake_fit_sir(y):
        return SIRFit(beta=0.5, gamma=0.5, R0=1.0, rmse=0.0, converged=False, n_iter=0)

    monkeypatch.setattr(sir_eval_mod, "fit_sir", _fake_fit_sir)

    out = sir_causal_forecast(daily_y_upto_w, (1, 2, 4), min_hist_weeks=4)

    assert out is None


def test_sir_causal_forecast_k_order_matches_horizon():
    """k=1 is the first future week (closest to w); later k's should not be
    identical to k=1 for a non-trivial (non-flat) SIR curve."""
    daily_y_upto_w = _make_bump_y(n=150, peak_frac=0.3)

    out = sir_causal_forecast(daily_y_upto_w, (1, 2, 4), min_hist_weeks=4)

    assert out is not None
    # Decaying-tail bump -> forecast values should differ across horizons.
    assert not (out[1] == out[2] == out[4])


def _origin_week_for_date(date: pd.Timestamp) -> int:
    iso = date.isocalendar()
    return (iso.year - 2017) * 52 + (iso.week - 1)


def _build_test_span_ts_df(song_id: str, chart: str, n_days: int, start: str) -> pd.DataFrame:
    """Daily series of length n_days starting at `start`, with values shaped like
    a clear bump so SIR converges; columns song_id, chart, date, y."""
    y = _make_bump_y(n=n_days)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    return pd.DataFrame({"song_id": song_id, "chart": chart, "date": dates, "y": y})


def test_run_sir_mode2_returns_expected_columns_and_test_span_only():
    # Origin week 0's date: 2017-01-02 (Monday) maps to week 0 per the ISO scheme.
    # TEST_START_WEEK (208) corresponds to 2020-12-28 onward (see dataset.py).
    ts_df = _build_test_span_ts_df("songA", "top200", n_days=400, start="2020-06-01")

    df = ts_df.sort_values("date")
    df["week"] = df["date"].apply(_origin_week_for_date)
    candidate_weeks = sorted(df.loc[df["week"] >= TEST_START_WEEK, "week"].unique().tolist())
    assert len(candidate_weeks) >= 3, "fixture must produce origins in the test span"

    origins = pd.DataFrame(
        {
            "song_id": ["songA"] * len(candidate_weeks),
            "chart": ["top200"] * len(candidate_weeks),
            "week": candidate_weeks,
        }
    )

    out = run_sir_mode2(ts_df, origins, ks=(1, 2, 4))

    assert list(out.columns) == ["song_id", "chart", "origin_week", "k", "y_pred_sir", "converged"]
    assert set(out["k"].unique()) <= {1, 2, 4}
    assert (out["origin_week"] >= TEST_START_WEEK).all()
    assert len(out) == len(candidate_weeks) * 3


def test_run_sir_mode2_filters_origins_outside_test_span():
    """Origins before TEST_START_WEEK must be dropped even if the caller forgot."""
    ts_df = _build_test_span_ts_df("songA", "top200", n_days=400, start="2020-06-01")

    df = ts_df.sort_values("date")
    df["week"] = df["date"].apply(_origin_week_for_date)
    in_span = sorted(df.loc[df["week"] >= TEST_START_WEEK, "week"].unique().tolist())
    out_of_span = sorted(df.loc[df["week"] < TEST_START_WEEK, "week"].unique().tolist())
    assert in_span and out_of_span

    origins = pd.DataFrame(
        {
            "song_id": ["songA"] * (len(in_span) + len(out_of_span)),
            "chart": ["top200"] * (len(in_span) + len(out_of_span)),
            "week": out_of_span + in_span,
        }
    )

    out = run_sir_mode2(ts_df, origins, ks=(1,))

    assert (out["origin_week"] >= TEST_START_WEEK).all()
    assert set(out["origin_week"].unique()) == set(in_span)


def test_run_sir_mode2_short_origin_emits_nan_row_not_dropped():
    """An origin too close to the start of the series (insufficient causal
    history) must still emit a row with converged=False, y_pred_sir=NaN."""
    # Build a series whose very first in-test-span week has < 4 weeks of history.
    # 2020-12-28 is the calendar start of ISO week 208; starting 10 days earlier
    # leaves only ~3.4 weeks of on-chart history by the time week 208 begins.
    start = "2020-12-18"
    ts_df = _build_test_span_ts_df("songA", "top200", n_days=30, start=start)

    df = ts_df.sort_values("date")
    df["week"] = df["date"].apply(_origin_week_for_date)
    in_span_weeks = sorted(df.loc[df["week"] >= TEST_START_WEEK, "week"].unique().tolist())
    assert in_span_weeks, "fixture must include at least one in-test-span origin"
    first_week = in_span_weeks[0]

    origins = pd.DataFrame({"song_id": ["songA"], "chart": ["top200"], "week": [first_week]})

    out = run_sir_mode2(ts_df, origins, ks=(1, 2, 4))

    assert len(out) == 3
    assert (~out["converged"]).all()
    assert out["y_pred_sir"].isna().all()


def test_run_sir_mode2_parallel_matches_sequential(monkeypatch):
    """Determinism: forcing n_jobs=1 (sequential) must give identical results to
    the default n_jobs=-1 (parallel) joblib execution."""
    import music_diffusion_gnn.evaluation.sir_eval as sir_eval_mod

    ts_df = pd.concat(
        [
            _build_test_span_ts_df("songA", "top200", n_days=200, start="2020-08-01"),
            _build_test_span_ts_df("songB", "viral50", n_days=180, start="2020-09-01"),
        ],
        ignore_index=True,
    )

    df = ts_df.sort_values("date")
    df["week"] = df["date"].apply(_origin_week_for_date)
    origins = (
        df[df["week"] >= TEST_START_WEEK][["song_id", "chart", "week"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    assert len(origins) > 0

    out_parallel = run_sir_mode2(ts_df, origins, ks=(1, 2, 4))

    real_parallel = sir_eval_mod.Parallel

    def _sequential_parallel(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return real_parallel(*args, **kwargs)

    monkeypatch.setattr(sir_eval_mod, "Parallel", _sequential_parallel)
    out_sequential = run_sir_mode2(ts_df, origins, ks=(1, 2, 4))

    out_parallel_sorted = out_parallel.sort_values(["song_id", "chart", "origin_week", "k"]).reset_index(
        drop=True
    )
    out_sequential_sorted = out_sequential.sort_values(
        ["song_id", "chart", "origin_week", "k"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(out_parallel_sorted, out_sequential_sorted)
