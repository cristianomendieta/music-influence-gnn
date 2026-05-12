"""Unit tests for evaluation.metrics."""
import numpy as np
import pytest

from music_diffusion_gnn.evaluation.metrics import mann_whitney_pairwise, rmse


def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert abs(rmse(y_true, y_pred) - 1.0) < 1e-9


def test_mann_whitney_different_distributions():
    rng = np.random.default_rng(42)
    a = rng.normal(0.03, 0.005, 1179)
    b = rng.normal(0.06, 0.01, 1179)
    result = mann_whitney_pairwise(a, b)
    assert result["p_value"] < 0.001, "Should detect difference between clearly distinct distributions"


def test_mann_whitney_same_distribution():
    rng = np.random.default_rng(42)
    a = rng.normal(0.05, 0.01, 1000)
    b = rng.normal(0.05, 0.01, 1000)
    result = mann_whitney_pairwise(a, b)
    assert result["p_value"] > 0.01, "Should not detect difference in identical distributions"


def test_mann_whitney_returns_dict():
    result = mann_whitney_pairwise(np.ones(50), np.zeros(50))
    assert "statistic" in result
    assert "p_value" in result
