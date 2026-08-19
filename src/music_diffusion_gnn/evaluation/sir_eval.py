"""SIR baseline evaluation — Mode-1 weekly curve reconstruction from Phase-0 fits,
Mode-2 causal refit+simulate forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from music_diffusion_gnn.baselines.sir import _sir_curve, fit_sir
from music_diffusion_gnn.training.dataset import get_split_regime


def sir_weekly_from_fit(daily_y: pd.Series, fit) -> pd.Series:
    """Reconstruct a song's daily SIR curve from its fit and aggregate to weekly.

    Replays `_sir_curve` over `t = 0..len(daily_y)-1` using `fit.beta`/`fit.gamma`
    and `I0 = daily_y.iloc[0]` (the same value `fit_sir` was originally given),
    then aggregates with the exact `aggregate_weekly` ISO-week rule (vectorized
    `dt.isocalendar()`, week = (iso_year-2017)*52 + (iso_week-1), kept in [0,260])
    so the SIR and GNN predictions land on an identical week axis.

    Args:
        daily_y: per-(song,chart) daily series, indexed by `date` (DatetimeIndex),
            sorted ascending by date.
        fit: object/Series with `.beta` and `.gamma` attributes (e.g. SIRFit, or a
            row of `sir_params_df`).

    Returns:
        pd.Series indexed by `week` (int, ascending), values = weekly mean of the
        reconstructed daily curve.
    """
    I0 = float(daily_y.iloc[0])
    t = pd.RangeIndex(len(daily_y)).to_numpy()
    curve = _sir_curve(t, fit.beta, fit.gamma, I0)

    df = pd.DataFrame({"date": pd.DatetimeIndex(daily_y.index), "y": curve})
    iso = df["date"].dt.isocalendar()
    df["week"] = (iso["year"].astype(int) - 2017) * 52 + (iso["week"].astype(int) - 1)
    df = df[(df["week"] >= 0) & (df["week"] <= 260)]

    return df.groupby("week")["y"].mean().sort_index()


def run_sir_mode1(ts_df: pd.DataFrame, sir_params_df: pd.DataFrame) -> pd.DataFrame:
    """Run SIR Mode-1 (free weekly curve from the Phase-0 fit) over a subset.

    For every (song_id, chart) present in both `ts_df` and `sir_params_df` (inner
    join — groups missing a fit are skipped, not raised), reconstructs and
    weekly-aggregates the SIR curve via `sir_weekly_from_fit`.

    Args:
        ts_df: daily timeseries with columns song_id, chart, date, y (and others,
            ignored).
        sir_params_df: indexed by (song_id, chart), columns beta, gamma, R0, rmse,
            converged, n_iter (Phase-0 `fit_all` output, e.g. `sir_params.parquet`).

    Returns:
        Long-format DataFrame, columns [song_id, chart, week, y_pred_sir], one row
        per (song, chart, week).
    """
    rows = []
    df = ts_df.sort_values(["song_id", "chart", "date"])
    for (song_id, chart), grp in df.groupby(["song_id", "chart"], observed=True):
        if (song_id, chart) not in sir_params_df.index:
            continue
        fit_row = sir_params_df.loc[(song_id, chart)]
        daily_y = pd.Series(grp["y"].to_numpy(), index=pd.DatetimeIndex(grp["date"]))
        weekly = sir_weekly_from_fit(daily_y, fit_row)
        for week, y_pred_sir in weekly.items():
            rows.append(
                {"song_id": song_id, "chart": chart, "week": int(week), "y_pred_sir": y_pred_sir}
            )

    return pd.DataFrame(rows, columns=["song_id", "chart", "week", "y_pred_sir"])


def sir_causal_forecast(
    daily_y_upto_w: np.ndarray, k_weeks: tuple[int, ...], *, min_hist_weeks: int = 4
) -> dict[int, float] | None:
    """Refit SIR causally on data `<= w` and simulate forward to `k` weeks ahead.

    Refits `fit_sir` on `daily_y_upto_w` only (no future data), then simulates the
    fitted curve out to `len(daily_y_upto_w) + 7*max(k_weeks)` days and aggregates
    the trailing `7*max(k_weeks)` days into one mean-per-7-day-block per horizon
    (k=1 is the first future week immediately after w, ..., k=max(k_weeks) the
    last). Simulated once for all horizons, then sliced.

    Args:
        daily_y_upto_w: daily series sorted ascending, values in [0, 0.5], strictly
            up to and including the origin week w.
        k_weeks: horizons (in weeks) to forecast, e.g. (1, 2, 4).
        min_hist_weeks: minimum weeks of on-chart history (`y > 0` days / 7)
            required to attempt a fit.

    Returns:
        {k: y_pred_sir} for each k in k_weeks, or None if there isn't enough
        on-chart history or the refit didn't converge.
    """
    n_onchart_days = int((daily_y_upto_w > 0).sum())
    if n_onchart_days / 7 < min_hist_weeks:
        return None

    fit = fit_sir(daily_y_upto_w)
    if not fit.converged:
        return None

    k_max = max(k_weeks)
    n = len(daily_y_upto_w)
    t = np.arange(n + 7 * k_max, dtype=float)
    I0 = float(daily_y_upto_w[0])
    curve = _sir_curve(t, fit.beta, fit.gamma, I0)

    future_weekly = curve[-7 * k_max :].reshape(k_max, 7).mean(axis=1)
    return {k: float(future_weekly[k - 1]) for k in k_weeks}


def _sir_mode2_one_group(
    song_id: str,
    chart: str,
    grp: pd.DataFrame,
    origin_weeks: list[int],
    ks: tuple[int, ...],
) -> list[dict]:
    """Build the (date,y,week) frame for one (song,chart) once, then forecast every
    origin week in `origin_weeks` for that group."""
    df = grp.sort_values("date")
    iso = pd.DatetimeIndex(df["date"]).isocalendar()
    week = (iso["year"].to_numpy().astype(int) - 2017) * 52 + (
        iso["week"].to_numpy().astype(int) - 1
    )
    y = df["y"].to_numpy()

    rows = []
    for w in origin_weeks:
        daily_y_upto_w = y[week <= w]
        forecast = sir_causal_forecast(daily_y_upto_w, ks, min_hist_weeks=4)
        for k in ks:
            if forecast is None:
                rows.append(
                    {
                        "song_id": song_id,
                        "chart": chart,
                        "origin_week": int(w),
                        "k": int(k),
                        "y_pred_sir": float("nan"),
                        "converged": False,
                    }
                )
            else:
                rows.append(
                    {
                        "song_id": song_id,
                        "chart": chart,
                        "origin_week": int(w),
                        "k": int(k),
                        "y_pred_sir": forecast[k],
                        "converged": True,
                    }
                )
    return rows


def run_sir_mode2(
    ts_df: pd.DataFrame,
    origins: pd.DataFrame,
    ks: tuple[int, ...] = (1, 2, 4),
    regime: str = "current",
) -> pd.DataFrame:
    """Run SIR Mode-2 (causal refit-and-simulate per origin) over a set of origins.

    Restricts `origins` to the `regime`'s test span as a safety net even if
    the caller already filtered. For each (song_id, chart) group, builds its
    (date, y, week) frame once, then forecasts every origin week for that
    group via `sir_causal_forecast`. Non-convergent or too-short origins still
    emit a row per k with `y_pred_sir=NaN, converged=False` — never silently
    dropped. Parallelized per (song_id, chart) group via joblib (loky backend).

    Args:
        ts_df: daily timeseries with columns song_id, chart, date, y (and others,
            ignored).
        origins: DataFrame with columns [song_id, chart, week] — one row per
            evaluation origin.
        ks: forecast horizons in weeks.
        regime: split-regime name (`SPLIT_REGIMES`, default "current") whose
            test-span boundaries gate `origins`.

    Returns:
        DataFrame, columns exactly [song_id, chart, origin_week, k, y_pred_sir,
        converged], one row per (origin, k).
    """
    r = get_split_regime(regime)
    valid = origins["week"] >= r.test_start_week
    if r.test_end_week is not None:
        valid &= origins["week"] <= r.test_end_week
    origins = origins[valid]

    origins_by_group = origins.groupby(["song_id", "chart"], observed=True)["week"].apply(
        lambda s: sorted(s.unique().tolist())
    )

    ts_groups = {key: grp for key, grp in ts_df.groupby(["song_id", "chart"], observed=True)}

    tasks = []
    for (song_id, chart), origin_weeks in origins_by_group.items():
        grp = ts_groups.get((song_id, chart))
        if grp is None:
            continue
        tasks.append(
            delayed(_sir_mode2_one_group)(song_id, chart, grp, origin_weeks, ks)
        )

    results = Parallel(n_jobs=-1, backend="loky")(tasks)

    rows: list[dict] = []
    for group_rows in results:
        rows.extend(group_rows)

    columns = ["song_id", "chart", "origin_week", "k", "y_pred_sir", "converged"]
    return pd.DataFrame(rows, columns=columns)
