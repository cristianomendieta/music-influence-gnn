"""Classic SIR baseline — per-song fit via odeint + curve_fit.

Implements Oliveira et al. BraSNAM 2025, Section 4.2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit


@dataclass
class SIRFit:
    beta: float
    gamma: float
    R0: float
    rmse: float
    converged: bool
    n_iter: int


def _sir_ode(state: list[float], _t: float, beta: float, gamma: float) -> list[float]:
    S, I, R = state
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def _sir_curve(t: np.ndarray, beta: float, gamma: float, I0: float) -> np.ndarray:
    S0 = max(1.0 - I0, 1e-6)
    y0 = [S0, I0, 0.0]
    try:
        sol = odeint(_sir_ode, y0, t, args=(beta, gamma), rtol=1e-4, atol=1e-6)
        return sol[:, 1]
    except Exception:
        return np.full(len(t), I0)


def fit_sir(y: np.ndarray, t: np.ndarray | None = None) -> SIRFit:
    """Fit a classic SIR model to observed series y.

    Parameters
    ----------
    y : 1-D array, normalized infectious proxy (values in [0, 0.5]).
    t : optional time array (defaults to 0, 1, ..., len(y)-1).

    Returns
    -------
    SIRFit with beta, gamma, R0, rmse, converged flag, and nfev.
    """
    if t is None:
        t = np.arange(len(y), dtype=float)

    I0 = float(np.clip(y[0], 1e-6, 1.0 - 1e-6))

    def model(t_arr: np.ndarray, beta: float, gamma: float) -> np.ndarray:
        return _sir_curve(t_arr, beta, gamma, I0)

    converged = False
    n_iter = 0
    beta, gamma = 0.5, 0.5

    try:
        popt, _, info, _, ier = curve_fit(
            model,
            t,
            y,
            p0=[0.5, 0.5],
            bounds=([0.0, 0.0], [10.0, 10.0]),
            full_output=True,
            maxfev=5000,
        )
        beta, gamma = float(popt[0]), float(popt[1])
        converged = ier in (1, 2, 3, 4)
        n_iter = int(info["nfev"])
    except (RuntimeError, ValueError):
        n_iter = 0

    y_pred = model(t, beta, gamma)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    R0 = beta / gamma if gamma > 1e-9 else float("inf")

    return SIRFit(beta=beta, gamma=gamma, R0=R0, rmse=rmse, converged=converged, n_iter=n_iter)
