"""Temporal helpers — canonical week calendar and temporal edge masking."""
from __future__ import annotations

from datetime import date

import torch
from torch_geometric.data import HeteroData

_WEEK_MIN = 0    # 2017-W1
_WEEK_MAX = 260  # 2021-W52


def week_index(d: date | str) -> int:
    """Return ISO-week offset from 2017-W1. Range: [0, 260].

    Mapping: (year - 2017) * 52 + (iso_week - 1).
    Raises ValueError for dates outside 2017-W1 .. 2021-W52.
    """
    if isinstance(d, str):
        d = date.fromisoformat(d)
    iso = d.isocalendar()
    year, week = iso[0], iso[1]
    idx = (year - 2017) * 52 + (week - 1)
    if idx < _WEEK_MIN or idx > _WEEK_MAX:
        raise ValueError(
            f"Date {d} (ISO {year}-W{week:02d}) maps to index {idx}, "
            f"outside allowed range [{_WEEK_MIN}, {_WEEK_MAX}]."
        )
    return idx


def mask_until(hetero: HeteroData, week_t: int) -> HeteroData:
    """Return shallow clone of hetero with only edges where first_seen_week <= week_t.

    Node feature tensors are shared (no deep copy).
    Accepts two layouts per edge type:
      - edge_attr exists: first_seen_week is edge_attr[:, -1]
      - no edge_attr: first_seen_week is a separate tensor attribute
    """
    if week_t < _WEEK_MIN or week_t > _WEEK_MAX:
        raise ValueError(f"week_t must be in [{_WEEK_MIN}, {_WEEK_MAX}], got {week_t}")

    g = HeteroData()

    # Shallow copy node attributes (shared tensors)
    for ntype in hetero.node_types:
        for key, val in hetero[ntype].items():
            g[ntype][key] = val

    # Filter edges by first_seen_week
    for edge_type in hetero.edge_types:
        et = hetero[edge_type]
        keys = set(et.keys())

        has_edge_attr = "edge_attr" in keys and et.edge_attr is not None
        has_fsw = "first_seen_week" in keys

        if has_edge_attr:
            fsw = et.edge_attr[:, -1]
        elif has_fsw:
            fsw = et.first_seen_week
        else:
            # No temporal info: copy as-is
            for key, val in et.items():
                g[edge_type][key] = val
            continue

        mask = fsw <= week_t
        g[edge_type].edge_index = et.edge_index[:, mask]
        if has_edge_attr:
            g[edge_type].edge_attr = et.edge_attr[mask]
        if has_fsw:
            g[edge_type].first_seen_week = et.first_seen_week[mask]

    return g
