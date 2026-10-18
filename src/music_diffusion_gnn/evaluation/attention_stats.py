"""Entropy of attention by in-degree (Phase 7, ATT-06; descriptive only).

Pure numpy/pandas over the output of ``return_attention=True`` of
``models.attention_encoder.HeteroAttentionEncoder``: ``(edge_index, alpha)`` with
``alpha`` of shape ``(E, heads)``. Attention is a description of the learned
weights, never an explanation of the prediction (Jain & Wallace, 2019, ADR-0006);
the causal measure stays the edge-type ablation.

Conventions:

- The in-degree of a destination node is its number of incoming edges, parallel
  edges (a pair on both charts) counted separately, as the encoder attends to
  each one.
- The entropy is normalised by ``log(d)`` so a uniform distribution over ``d``
  neighbours gives 1 and a single dominant neighbour gives 0.
- **Degree 1 gives 0 by convention**: ``log(1) = 0`` cannot normalise, and with
  one neighbour the softmax is fixed at 1. Such nodes carry no choice, so they
  get their own bin in :func:`entropy_by_degree` rather than diluting the others.
- Degree 0 gives NaN (no attention distribution exists).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Lower edges of the in-degree bins; the last bin is open-ended.
DEFAULT_BINS = (1, 2, 4, 8, 16, 32, 64, 128)


def normalized_attention_entropy(
    edge_index: np.ndarray, alpha: np.ndarray, n_dst: int
) -> np.ndarray:
    """Normalised entropy of the attention distribution of each destination node.

    Args:
        edge_index: ``(2, E)``, destination node in row 1
        alpha: ``(E,)`` or ``(E, heads)`` attention weights per edge
        n_dst: number of destination nodes

    Returns:
        ``(n_dst,)`` (or ``(n_dst, heads)``) in ``[0, 1]``: ``H / log(d)`` with ``d``
        the in-degree. 0 for ``d == 1`` (convention, see module docstring), NaN
        for ``d == 0``. The weights of each node are renormalised to sum to 1,
        so the result does not depend on edge order or on a global scale.
    """
    dst = np.asarray(edge_index)[1].astype(np.int64)
    alpha = np.asarray(alpha, dtype=np.float64)
    squeeze = alpha.ndim == 1
    if squeeze:
        alpha = alpha[:, None]
    deg = np.bincount(dst, minlength=n_dst)

    out = np.empty((n_dst, alpha.shape[1]))
    for h in range(alpha.shape[1]):
        a = alpha[:, h]
        total = np.bincount(dst, weights=a, minlength=n_dst)
        p = np.divide(a, total[dst], out=np.zeros_like(a), where=total[dst] > 0)
        plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
        entropy = -np.bincount(dst, weights=plogp, minlength=n_dst)
        norm = np.log(np.maximum(deg, 2))   # d=0 and d=1 are overwritten below
        col = entropy / norm
        col[deg == 1] = 0.0
        col[deg == 0] = np.nan
        out[:, h] = np.clip(col, 0.0, 1.0)
    return out[:, 0] if squeeze else out


def entropy_by_degree(
    edge_index: np.ndarray,
    alpha: np.ndarray,
    n_dst: int,
    bins: tuple[int, ...] = DEFAULT_BINS,
) -> pd.DataFrame:
    """Aggregate the node entropy by in-degree bin (one row per bin).

    Heads are averaged within each node first. ``bins`` holds the lower edges of
    the bins; the last bin is open-ended. Nodes with no incoming edge are left
    out, so ``n_nodes`` sums to the number of nodes with at least one edge. Empty
    bins stay in the table (``n_nodes = 0``, NaN entropy) so two tables, e.g.
    ``A`` and ``A-emb``, line up row by row.

    Returns:
        DataFrame with ``degree_bin`` (label), ``degree_min``, ``degree_max``
        (NaN for the open bin), ``n_nodes``, ``entropy_mean``, ``entropy_median``.
    """
    ent = normalized_attention_entropy(edge_index, alpha, n_dst)
    if ent.ndim == 2:
        ent = ent.mean(axis=1)
    deg = np.bincount(np.asarray(edge_index)[1].astype(np.int64), minlength=n_dst)
    has_edge = deg >= 1

    lows = list(bins)
    highs = [b - 1 for b in lows[1:]] + [np.nan]
    rows = []
    for lo, hi in zip(lows, highs):
        sel = has_edge & (deg >= lo) & (np.isnan(hi) | (deg <= hi))
        values = ent[sel]
        if np.isnan(hi):
            label = f"{lo}+"
        else:
            label = str(lo) if lo == hi else f"{lo}-{int(hi)}"
        rows.append({
            "degree_bin": label,
            "degree_min": lo,
            "degree_max": hi,
            "n_nodes": int(sel.sum()),
            "entropy_mean": float(values.mean()) if len(values) else np.nan,
            "entropy_median": float(np.median(values)) if len(values) else np.nan,
        })
    return pd.DataFrame(rows)
