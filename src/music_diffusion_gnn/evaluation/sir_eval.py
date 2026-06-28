"""SIR baseline evaluation — Mode-1 weekly curve reconstruction from Phase-0 fits."""
from __future__ import annotations

import pandas as pd

from music_diffusion_gnn.baselines.sir import _sir_curve


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
