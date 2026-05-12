"""Generic parallel runner for per-song baseline fits using joblib."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _fit_one(
    song_id: str,
    chart: str,
    y: np.ndarray,
    fit_fn: Callable[[np.ndarray], Any],
) -> dict[str, Any]:
    result = fit_fn(y)
    row: dict[str, Any] = {"song_id": song_id, "chart": chart}
    row.update(result.__dict__)
    return row


def fit_all(
    timeseries: pd.DataFrame,
    fit_fn: Callable[[np.ndarray], Any],
    *,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Apply fit_fn to each (song_id, chart) group in timeseries.

    Parameters
    ----------
    timeseries : long-format DataFrame with columns song_id, chart, date, y.
    fit_fn     : callable that takes a 1-D numpy array and returns a dataclass.
    n_jobs     : joblib parallelism (-1 = all cores).

    Returns
    -------
    DataFrame indexed by (song_id, chart) with one row per fit.
    """
    groups = list(timeseries.sort_values(["song_id", "chart", "date"])
                  .groupby(["song_id", "chart"]))

    tasks = [
        delayed(_fit_one)(song_id, chart, grp["y"].to_numpy(), fit_fn)
        for (song_id, chart), grp in groups
    ]

    rows = Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    df = pd.DataFrame(rows).set_index(["song_id", "chart"])
    return df
