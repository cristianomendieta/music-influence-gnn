"""Test SIR fit on synthetic curves — recovers known beta and gamma within 5%."""
import numpy as np
import pytest
from scipy.integrate import odeint

from music_diffusion_gnn.baselines.sir import SIRFit, _sir_ode, fit_sir


def _make_sir_series(beta: float, gamma: float, I0: float = 0.01, n: int = 200) -> np.ndarray:
    t = np.arange(n, dtype=float)
    y0 = [1.0 - I0, I0, 0.0]
    sol = odeint(_sir_ode, y0, t, args=(beta, gamma))
    return sol[:, 1]


def test_fit_recovers_beta_gamma():
    beta_true, gamma_true = 0.3, 0.1
    y = _make_sir_series(beta_true, gamma_true)
    result = fit_sir(y)
    assert result.converged, "Fit did not converge on clean synthetic data"
    assert abs(result.beta - beta_true) / beta_true < 0.05, f"beta off: {result.beta:.4f} vs {beta_true}"
    assert abs(result.gamma - gamma_true) / gamma_true < 0.05, f"gamma off: {result.gamma:.4f} vs {gamma_true}"


def test_r0_calculation():
    beta_true, gamma_true = 0.4, 0.2
    y = _make_sir_series(beta_true, gamma_true)
    result = fit_sir(y)
    expected_R0 = beta_true / gamma_true
    assert abs(result.R0 - expected_R0) / expected_R0 < 0.1, f"R0 off: {result.R0:.4f} vs {expected_R0}"


def test_rmse_low_on_clean_data():
    y = _make_sir_series(0.5, 0.15)
    result = fit_sir(y)
    assert result.rmse < 0.005, f"RMSE too high on clean data: {result.rmse:.6f}"


def test_fit_returns_sirfit_type():
    y = _make_sir_series(0.3, 0.1)
    result = fit_sir(y)
    assert isinstance(result, SIRFit)


def test_custom_time_array():
    t = np.linspace(0, 200, 100)
    beta_true, gamma_true = 0.3, 0.1
    y0 = [1.0 - 0.01, 0.01, 0.0]
    sol = odeint(_sir_ode, y0, t, args=(beta_true, gamma_true))
    y = sol[:, 1]
    result = fit_sir(y, t=t)
    assert result.converged
    assert result.rmse < 0.01


def test_flat_series_does_not_crash():
    y = np.full(100, 0.001)
    result = fit_sir(y)
    assert isinstance(result, SIRFit)
