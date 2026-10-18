"""Unit tests for evaluation/attention_stats.py (Phase 7, ATT-06, T13).

Hand-computed attention on tiny graphs (numpy/pandas only, no torch).
``edge_index`` is ``(2, E)`` with the destination in row 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from music_diffusion_gnn.evaluation.attention_stats import (
    entropy_by_degree,
    normalized_attention_entropy,
)


def _star(alpha):
    """All ``len(alpha)`` edges arrive at destination node 0 of a 3-node graph."""
    d = len(alpha)
    edge_index = np.stack([np.arange(1, d + 1), np.zeros(d, dtype=int)])
    return edge_index, np.asarray(alpha, dtype=float)


@pytest.mark.parametrize("d", [2, 5, 40])
def test_uniform_attention_on_d_neighbours_has_entropy_one(d):
    edge_index, alpha = _star(np.full(d, 1.0 / d))
    h = normalized_attention_entropy(edge_index, alpha, n_dst=1)
    assert h[0] == pytest.approx(1.0)


def test_attention_on_a_single_neighbour_has_entropy_zero():
    edge_index, alpha = _star([1.0, 0.0, 0.0, 0.0])
    h = normalized_attention_entropy(edge_index, alpha, n_dst=1)
    assert h[0] == pytest.approx(0.0)


def test_entropy_matches_hand_value_for_a_skewed_distribution():
    # H([.5, .25, .25]) = 1.5 ln 2 ; normalised by ln 3
    edge_index, alpha = _star([0.5, 0.25, 0.25])
    h = normalized_attention_entropy(edge_index, alpha, n_dst=1)
    assert h[0] == pytest.approx(1.5 * np.log(2) / np.log(3))


def test_entropy_does_not_depend_on_the_scale_of_the_weights():
    # Docstring promise: weights are renormalised per destination. Node 0 gets
    # [.5, .25, .25] x 3 (sums to 3), node 1 gets [1, 1] x 0.2 (sums to 0.4).
    edge_index = np.array([[2, 3, 4, 2, 3], [0, 0, 0, 1, 1]])
    normalised = np.array([0.5, 0.25, 0.25, 0.5, 0.5])
    scaled = normalised * np.array([3.0, 3.0, 3.0, 0.4, 0.4])
    h = normalized_attention_entropy(edge_index, scaled, n_dst=2)
    assert h == pytest.approx(normalized_attention_entropy(edge_index, normalised, n_dst=2))
    assert h == pytest.approx([1.5 * np.log(2) / np.log(3), 1.0])


def test_in_degree_one_is_zero_by_convention_and_in_degree_zero_is_nan():
    # node 0 has one incoming edge (degree 1), node 1 has none, node 2 has two
    edge_index = np.array([[1, 0, 1], [0, 2, 2]])
    alpha = np.array([1.0, 0.5, 0.5])
    h = normalized_attention_entropy(edge_index, alpha, n_dst=3)
    assert h[0] == 0.0
    assert np.isnan(h[1])
    assert h[2] == pytest.approx(1.0)


def test_parallel_edges_count_as_separate_neighbours():
    # two parallel edges (same pair, one per chart) with equal weight: d = 2
    edge_index = np.array([[1, 1], [0, 0]])
    h = normalized_attention_entropy(edge_index, np.array([0.5, 0.5]), n_dst=1)
    assert h[0] == pytest.approx(1.0)


def test_entropy_is_invariant_to_edge_order():
    rng = np.random.default_rng(0)
    n_dst, n_edges = 6, 40
    dst = rng.integers(0, n_dst, n_edges)
    src = rng.integers(0, 10, n_edges)
    raw = rng.random(n_edges)
    # normalise per destination so the weights are a real attention distribution
    alpha = raw / np.bincount(dst, weights=raw, minlength=n_dst)[dst]
    edge_index = np.stack([src, dst])

    base = normalized_attention_entropy(edge_index, alpha, n_dst)
    perm = rng.permutation(n_edges)
    shuffled = normalized_attention_entropy(edge_index[:, perm], alpha[perm], n_dst)
    np.testing.assert_allclose(base, shuffled, equal_nan=True)


def test_one_entropy_per_head_when_alpha_has_a_head_axis():
    edge_index, _ = _star([0.5, 0.5])
    alpha = np.array([[0.5, 1.0], [0.5, 0.0]])   # head 0 uniform, head 1 concentrated
    h = normalized_attention_entropy(edge_index, alpha, n_dst=1)
    assert h.shape == (1, 2)
    np.testing.assert_allclose(h[0], [1.0, 0.0], atol=1e-12)


def test_entropy_by_degree_has_one_row_per_bin_and_counts_nodes_with_edges():
    # in-degrees: node0=1, node1=2, node2=3, node3=5, node4=0 (isolated)
    degrees = {0: 1, 1: 2, 2: 3, 3: 5}
    src, dst, alpha = [], [], []
    for node, d in degrees.items():
        src += [9] * d
        dst += [node] * d
        alpha += [1.0 / d] * d
    table = entropy_by_degree(np.array([src, dst]), np.array(alpha), n_dst=5, bins=(1, 2, 4))

    assert len(table) == 3                                   # bins [1,2) [2,4) [4,inf)
    assert table["n_nodes"].tolist() == [1, 2, 1]            # node4 (degree 0) is out
    assert table["n_nodes"].sum() == 4                       # nodes with >= 1 edge


def test_entropy_by_degree_averages_entropy_within_the_bin():
    # node0: d=2 uniform (H=1); node1: d=3 fully concentrated (H=0). Both in bin [2,4).
    edge_index = np.array([[7, 7, 7, 7, 7], [0, 0, 1, 1, 1]])
    alpha = np.array([0.5, 0.5, 1.0, 0.0, 0.0])
    table = entropy_by_degree(edge_index, alpha, n_dst=2, bins=(1, 2, 4))
    row = table.iloc[1]
    assert row["n_nodes"] == 2
    assert row["entropy_mean"] == pytest.approx(0.5)


def test_entropy_by_degree_averages_heads_per_node():
    # one node, d=2: head 0 uniform (1.0), head 1 concentrated (0.0) -> node entropy 0.5
    edge_index = np.array([[1, 2], [0, 0]])
    alpha = np.array([[0.5, 1.0], [0.5, 0.0]])
    table = entropy_by_degree(edge_index, alpha, n_dst=1, bins=(1, 2, 4))
    assert table.iloc[1]["entropy_mean"] == pytest.approx(0.5)
