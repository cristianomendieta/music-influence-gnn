"""Naive baselines for popularity forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_predict(
    y_week_df: pd.DataFrame,
    split: pd.DataFrame,
) -> np.ndarray:
    """Naive persistence baseline: ŷ(w) = y(w-1).

    For each (song_id, chart, target_week) tuple in ``split``, returns the
    observed ``y_week`` from the *previous* week. When the previous week has
    no observation (first week of the span), uses the minimum floor value
    ``0.0`` to avoid future leakage.

    Args:
        y_week_df: the full weekly DataFrame (all splits combined) with
            columns ``[song_id, chart, week, y_week]``.
        split: subset DataFrame (one of train/val/test) containing the
            target tuples to evaluate.

    Returns:
        Numpy array of shape ``(N,)`` aligned to the rows of ``split``,
        giving the persistence prediction for each row.
    """
    # Index for O(1) lookup
    indexed = y_week_df.set_index(["song_id", "chart", "week"])["y_week"]

    preds = []
    for _, row in split.iterrows():
        prev_week = row["week"] - 1
        key = (row["song_id"], row["chart"], prev_week)
        if key in indexed.index:
            preds.append(float(indexed[key]))
        else:
            preds.append(0.0)

    return np.array(preds, dtype=np.float32)


def persistence_predict_bulk(
    y_week_df: pd.DataFrame,
    split: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return persistence predictions split by chart regime.

    Returns dict with keys ``'viral50'`` and ``'top200'``.
    """
    result = {}
    for chart_name in ("viral50", "top200"):
        mask = split["chart"] == chart_name
        sub = split[mask].copy()
        if len(sub) == 0:
            result[chart_name] = np.array([], dtype=np.float32)
        else:
            result[chart_name] = persistence_predict(y_week_df, sub)
    return result
