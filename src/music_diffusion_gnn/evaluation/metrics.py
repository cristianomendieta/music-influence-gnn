"""Evaluation metrics: RMSE and Mann-Whitney paired test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mann_whitney_pairwise(
    rmse_a: np.ndarray | pd.Series,
    rmse_b: np.ndarray | pd.Series,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Mann-Whitney U test between two paired RMSE distributions.

    Returns dict with keys: statistic, p_value.
    """
    rmse_a = np.asarray(rmse_a, dtype=float)
    rmse_b = np.asarray(rmse_b, dtype=float)
    result = stats.mannwhitneyu(rmse_a, rmse_b, alternative=alternative)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def summarize_rmse(sir_rmse: pd.Series) -> pd.DataFrame:
    """Build a summary table of SIR RMSE per chart."""
    rows = []
    for chart in sir_rmse.index.get_level_values("chart").unique():
        sir_vals = sir_rmse.xs(chart, level="chart")
        rows.append({
            "chart": chart,
            "sir_mean": sir_vals.mean(),
            "sir_median": sir_vals.median(),
            "sir_std": sir_vals.std(),
        })
    return pd.DataFrame(rows).set_index("chart")
