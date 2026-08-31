"""Structural controls for the comparison ladder — no-graph and rewired-graph.

Two ablations of the *structure* that keep everything else fixed (same node
features, same architecture, same parameter budget, same training protocol):

``strip_all_edges``
    Every edge type is emptied. ``SAGEConv`` then reduces to its root branch
    (a per-node-type linear map), so the encoder degenerates into an MLP over
    node features and the model has no access to any relation. This is the
    neural-baseline-without-graph of issue 07.

``rewire_preserving_degree``
    Destinations are permuted **within each first-seen-week stratum** of each
    edge type. Both degree sequences and the whole temporal profile survive
    exactly; only *which* node connects to *which* changes. This is the
    shuffled-graph control of issue 08 / ADR-0002.

Neither function mutates its input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def _first_seen_week(store) -> torch.Tensor | None:
    """Return the per-edge first-seen week, whichever layout the store uses.

    Mirrors :func:`graph.temporal.mask_until`: ``edge_attr[:, -1]`` when there
    is an ``edge_attr``, otherwise the separate ``first_seen_week`` tensor.
    ``None`` means the edge type carries no temporal information.
    """
    keys = set(store.keys())
    if "edge_attr" in keys and store.edge_attr is not None:
        return store.edge_attr[:, -1]
    if "first_seen_week" in keys and store.first_seen_week is not None:
        return store.first_seen_week
    return None


def strip_all_edges(g: HeteroData) -> HeteroData:
    """Return a copy of ``g`` with every edge type emptied (issue 07).

    Node features, node counts and the edge-type schema are preserved, so the
    same model class instantiates with the same parameter count — the
    neighbour branch of each ``SAGEConv`` simply never receives a message.
    """
    out = g.clone()
    for edge_type in out.edge_types:
        store = out[edge_type]
        device = store.edge_index.device
        store.edge_index = torch.empty((2, 0), dtype=store.edge_index.dtype, device=device)
        if "edge_attr" in store and store.edge_attr is not None:
            store.edge_attr = store.edge_attr[:0]
        if "first_seen_week" in store and store.first_seen_week is not None:
            store.first_seen_week = store.first_seen_week[:0]
    return out


def _rewire_one(
    src: np.ndarray,
    dst: np.ndarray,
    fsw: np.ndarray | None,
    *,
    same_type: bool,
    rng: np.random.Generator,
    max_swap_attempts: int = 32,
) -> tuple[np.ndarray, int]:
    """Permute ``dst`` within each ``fsw`` stratum. Returns (new_dst, n_self_loops).

    Permuting inside a stratum keeps, for every week, the exact set of source
    nodes and the exact multiset of destination nodes that the real graph had
    at that week — so out-degree, in-degree and the per-week edge count are all
    preserved exactly, and no edge becomes active earlier than it really was.
    """
    new_dst = dst.copy()
    strata = [np.arange(len(dst))] if fsw is None else [
        np.flatnonzero(fsw == v) for v in np.unique(fsw)
    ]

    n_self = 0
    for idx in strata:
        if len(idx) < 2:
            continue
        permuted = dst[idx][rng.permutation(len(idx))]
        if same_type:
            # A self-loop is an artefact of the permutation, not of the data.
            # Swap each one with a random other position of the same stratum.
            # One pair at a time: a vectorised swap with overlapping index
            # arrays would drop a value and stop being a permutation, which is
            # exactly the in-degree guarantee this function sells.
            src_s = src[idx]
            for _ in range(max_swap_attempts):
                bad = np.flatnonzero(permuted == src_s)
                if len(bad) == 0:
                    break
                for i in bad:
                    j = int(rng.integers(0, len(idx)))
                    permuted[i], permuted[j] = permuted[j], permuted[i]
            n_self += int(np.count_nonzero(permuted == src_s))
        new_dst[idx] = permuted
    return new_dst, n_self


def rewire_preserving_degree(
    g: HeteroData,
    seed: int,
    edge_types: list[tuple[str, str, str]] | None = None,
) -> HeteroData:
    """Return a copy of ``g`` with edges randomly rewired (issue 08, ADR-0002).

    Args:
        g: the graph to rewire (never mutated)
        seed: seed of the permutation — the rewired graph is a function of it,
            so a run is reproducible from ``(graph, seed)`` alone
        edge_types: restrict the rewiring to these types (default: all)

    Preserved exactly, per edge type: number of edges, out-degree of every
    source node, in-degree multiset of destination nodes, and the
    first-seen-week of every edge (hence the number of active edges at every
    week). Destroyed: which specific pair each edge joins.

    ``edge_attr`` rows are left in place, aligned with their original edge, so
    the last column keeps being the correct first-seen week for masking. The
    remaining columns describe a pair that no longer exists; nothing in the
    encoder reads them (``SAGEConv`` ignores ``edge_attr``).
    """
    rng = np.random.default_rng(seed)
    out = g.clone()
    targets = list(out.edge_types) if edge_types is None else list(edge_types)

    for edge_type in targets:
        store = out[edge_type]
        ei = store.edge_index
        if ei.shape[1] == 0:
            continue
        fsw_t = _first_seen_week(store)
        src = ei[0].cpu().numpy()
        dst = ei[1].cpu().numpy()
        fsw = None if fsw_t is None else fsw_t.cpu().numpy()

        new_dst, _ = _rewire_one(
            src, dst, fsw, same_type=(edge_type[0] == edge_type[2]), rng=rng
        )
        store.edge_index = torch.stack([
            ei[0], torch.as_tensor(new_dst, dtype=ei.dtype, device=ei.device)
        ])
    return out


def edge_report(g: HeteroData) -> pd.DataFrame:
    """One row per edge type: counts, degree extremes and self-loops.

    Used by the notebooks to show, side by side, that a rewired graph matches
    the real one on everything except topology.
    """
    rows = []
    for edge_type in g.edge_types:
        ei = g[edge_type].edge_index
        n = ei.shape[1]
        src, dst = ei[0], ei[1]
        rows.append({
            "edge_type": str(edge_type),
            "n_edges": n,
            "n_src_distintos": int(torch.unique(src).numel()) if n else 0,
            "n_dst_distintos": int(torch.unique(dst).numel()) if n else 0,
            "grau_saida_max": int(torch.bincount(src).max()) if n else 0,
            "grau_entrada_max": int(torch.bincount(dst).max()) if n else 0,
            "self_loops": int((src == dst).sum()) if n else 0,
        })
    return pd.DataFrame(rows)
