"""Causal edge features for the attention encoder (ADR-0006).

Only attributes that are known at the snapshot week ``w`` reach the encoder.
The cotrajectory ``edge_attr`` layout is
``[days_together, avg_position_distance, chart, first_seen_week]``
(``graph/edges.py::build_cotrajectory``). The first two columns are summed over
the whole period, so they leak the future into the snapshot of ``w`` and are
dropped. What is left is the chart of the edge and its age at ``w``.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

_COTRAJ_ET = ("music", "cotrajectory", "music")
_CHART_COL = 2


def causal_edge_features(snap: HeteroData, week: int) -> dict[tuple, Tensor]:
    """Return ``{edge_type: (E, 2)}`` with ``[chart, log1p(week - first_seen_week)]``.

    Only ``(music, cotrajectory, music)`` has causal edge attributes; every
    other relation is absent from the result (the encoder reads them by content
    only). ``snap`` must be ``mask_until(g, week)``: an edge newer than
    ``week`` means the snapshot was not masked, and raises ``ValueError``.
    """
    out: dict[tuple, Tensor] = {}
    if _COTRAJ_ET not in snap.edge_types:
        return out
    attr = snap[_COTRAJ_ET].edge_attr
    first_seen = attr[:, -1]
    if (first_seen > week).any():
        raise ValueError(
            f"cotrajectory edge with first_seen_week {first_seen.max().item():g} > week {week}: "
            "pass mask_until(g, week), not the full graph"
        )
    age = torch.log1p(week - first_seen)
    out[_COTRAJ_ET] = torch.stack([attr[:, _CHART_COL], age], dim=1)
    return out
