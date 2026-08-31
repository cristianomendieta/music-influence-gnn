"""Structural controls: no-graph (issue 07) and rewired graph (issue 08)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from music_diffusion_gnn.graph.build import graph_path
from music_diffusion_gnn.graph.controls import (
    edge_report,
    rewire_preserving_degree,
    strip_all_edges,
)
from music_diffusion_gnn.graph.temporal import mask_until

GRAPH_DIR = "data/processed/graph"


@pytest.fixture(scope="module")
def g():
    path = graph_path("current", GRAPH_DIR)
    if not path.exists():
        pytest.skip(f"graph not built: {path}")
    return torch.load(path, weights_only=False)


def _degrees(store, n_src=None, n_dst=None):
    ei = store.edge_index
    return (
        torch.bincount(ei[0], minlength=n_src or 0),
        torch.bincount(ei[1], minlength=n_dst or 0),
    )


# --- issue 07: no edges at all -------------------------------------------

def test_strip_all_edges_leaves_no_edge(g):
    stripped = strip_all_edges(g)
    for et in stripped.edge_types:
        assert stripped[et].edge_index.shape == (2, 0), et
        if "edge_attr" in stripped[et] and stripped[et].edge_attr is not None:
            assert stripped[et].edge_attr.shape[0] == 0


def test_strip_all_edges_keeps_nodes_and_does_not_mutate(g):
    before = {et: g[et].edge_index.shape[1] for et in g.edge_types}
    stripped = strip_all_edges(g)
    for nt in g.node_types:
        assert stripped[nt].num_nodes == g[nt].num_nodes
        assert torch.equal(stripped[nt].x, g[nt].x)
    assert {et: g[et].edge_index.shape[1] for et in g.edge_types} == before


def test_stripped_graph_stays_empty_after_masking(g):
    """The model masks per week; an emptied graph must stay empty at any week."""
    snap = mask_until(strip_all_edges(g), 208)
    assert all(snap[et].edge_index.shape[1] == 0 for et in snap.edge_types)


# --- issue 08: rewired graph ---------------------------------------------

def test_rewire_preserves_edge_counts_and_degrees(g):
    shuffled = rewire_preserving_degree(g, seed=0)
    for et in g.edge_types:
        n_src = g[et[0]].num_nodes
        n_dst = g[et[2]].num_nodes
        assert shuffled[et].edge_index.shape == g[et].edge_index.shape, et
        out_real, in_real = _degrees(g[et], n_src, n_dst)
        out_shuf, in_shuf = _degrees(shuffled[et], n_src, n_dst)
        # Out-degree per node is untouched: the source row is never permuted.
        assert torch.equal(out_real, out_shuf), et
        # In-degree is preserved as a multiset (the same destinations, reassigned).
        assert torch.equal(torch.sort(in_real).values, torch.sort(in_shuf).values), et


def test_rewire_preserves_temporal_profile(g):
    """Same number of active edges at every week: no edge activates earlier."""
    shuffled = rewire_preserving_degree(g, seed=0)
    for week in (0, 60, 182, 208, 260):
        real, shuf = mask_until(g, week), mask_until(shuffled, week)
        for et in g.edge_types:
            assert real[et].edge_index.shape[1] == shuf[et].edge_index.shape[1], (et, week)


def test_rewire_actually_changes_topology(g):
    shuffled = rewire_preserving_degree(g, seed=0)
    et = ("music", "cotrajectory", "music")
    changed = (g[et].edge_index[1] != shuffled[et].edge_index[1]).float().mean()
    assert changed > 0.5, f"only {changed:.1%} of the destinations moved"


def test_rewire_creates_no_self_loops_on_same_type_edges(g):
    shuffled = rewire_preserving_degree(g, seed=0)
    for et in shuffled.edge_types:
        if et[0] != et[2]:
            continue
        ei = shuffled[et].edge_index
        assert int((ei[0] == ei[1]).sum()) == 0, et


def test_rewire_is_deterministic_and_does_not_mutate(g):
    before = {et: g[et].edge_index.clone() for et in g.edge_types}
    a = rewire_preserving_degree(g, seed=7)
    b = rewire_preserving_degree(g, seed=7)
    c = rewire_preserving_degree(g, seed=8)
    et = ("music", "cotrajectory", "music")
    assert torch.equal(a[et].edge_index, b[et].edge_index)
    assert not torch.equal(a[et].edge_index, c[et].edge_index)
    for edge_type, ei in before.items():
        assert torch.equal(g[edge_type].edge_index, ei), edge_type


def test_edge_report_matches_between_real_and_rewired(g):
    """Everything the report shows, except topology, has to agree."""
    real = edge_report(g).set_index("edge_type")
    shuf = edge_report(rewire_preserving_degree(g, seed=0)).set_index("edge_type")
    assert np.array_equal(real["n_edges"].values, shuf["n_edges"].values)
    assert np.array_equal(real["grau_saida_max"].values, shuf["grau_saida_max"].values)
