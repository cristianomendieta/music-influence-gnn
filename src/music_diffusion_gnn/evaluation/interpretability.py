"""Interpretability analyses for MusicDiffusionGNN (P2, R4): edge-type ablation,
feature-group permutation importance, and SIR-analogous rise/fall descriptive stats.

All three analyses share the output schema mandated by design.md §5 / R5.4
(``results/phase3/interpretability.parquet``): ``[analysis, component, delta_rmse,
regime]``. See ``population_analogs`` docstring for a SPEC_DEVIATION note on reusing
the ``delta_rmse`` column for non-RMSE values in that analysis.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from music_diffusion_gnn.evaluation.metrics import rmse

if TYPE_CHECKING:
    from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
    from music_diffusion_gnn.training.dataset import Sample


_SCHEMA = ["analysis", "component", "delta_rmse", "regime"]


def predict_all(
    model: "MusicDiffusionGNN",
    g: HeteroData,
    samples: list["Sample"],
    max_cotraj_edges: int | None = None,
) -> np.ndarray:
    """Run encode_weeks/predict per distinct target_week and return predictions
    aligned 1:1 with ``samples`` order. model.eval() + no_grad throughout.

    The bank must hold the *look-back window* weeks (``[w-W, ..., w-1]``), which
    is what :meth:`MusicDiffusionGNN.predict` gathers — not the target week ``w``
    itself. Encoding only ``[w]`` (the pre-2026-08 behaviour) left every window
    position missing from the bank, so ``predict`` fell back to a zero embedding
    for the whole sequence: Δ became a constant independent of the graph, and
    every ablation returned *exactly* zero delta_rmse. Embeddings are cached
    across target weeks since consecutive windows overlap heavily.
    """
    preds = np.empty(len(samples), dtype=np.float64)
    by_week: dict[int, list[int]] = {}
    for i, s in enumerate(samples):
        by_week.setdefault(s.target_week, []).append(i)

    model.eval()
    zcache: dict[int, Tensor] = {}
    with torch.no_grad():
        for week, idxs in by_week.items():
            # ``predict`` reads the bank week off ``samples[0].window_weeks[t]``, so a
            # first sample that debuted late (a ``-1`` at position t) blanks that
            # position for the WHOLE batch, and the graph stops reaching the GRU
            # there. Full week batches are unaffected (measured: the first sample
            # covers 100% of the test positions), but an ablation run on a subset
            # can silently measure nothing — the failure mode of issue 04, by a
            # different route. Putting the most complete window first is a no-op on
            # full batches and removes the order dependency on subsets.
            idxs = sorted(idxs, key=lambda i: sum(samples[i].pad_mask))
            week_samples = [samples[i] for i in idxs]
            # Union over the batch: padded positions (-1) differ per sample
            # (first_seen_week), the real weeks do not.
            window = sorted({w for s in week_samples for w in s.window_weeks if w >= 0})
            missing = [w for w in window if w not in zcache]
            if missing:
                zcache.update(model.encode_weeks(g, missing, max_cotraj_edges=max_cotraj_edges))
            bank = {w: zcache[w] for w in window}
            y_pred = model.predict(bank, week_samples)
            preds[idxs] = y_pred.detach().cpu().numpy()
    return preds


# Kept as a private alias: the corrected harness is referenced by this name in
# docs/diagnostico-ablacao.md and in the item-04 notebook.
_predict_all = predict_all


def _baseline_rmse(model: "MusicDiffusionGNN", g: HeteroData, samples: list["Sample"]) -> float:
    y_true = np.array([s.y for s in samples], dtype=np.float64)
    y_pred = predict_all(model, g, samples)
    return rmse(y_true, y_pred)


def _empty_edge_store(g: HeteroData, edge_type: tuple[str, str, str]) -> HeteroData:
    """Return a clone of g with edge_type's edges emptied (zero edges), all
    other edge types and node features intact. Does not mutate g."""
    g_ablated = g.clone()
    store = g_ablated[edge_type]
    device = store.edge_index.device
    store.edge_index = torch.empty((2, 0), dtype=store.edge_index.dtype, device=device)
    if "edge_attr" in store and store.edge_attr is not None:
        store.edge_attr = store.edge_attr[:0]
    if "first_seen_week" in store and store.first_seen_week is not None:
        store.first_seen_week = store.first_seen_week[:0]
    return g_ablated


def edge_type_ablation(
    model: "MusicDiffusionGNN",
    g: HeteroData,
    val_samples: list["Sample"],
) -> pd.DataFrame:
    """Delta RMSE from removing each edge type (entirely, zero edges) from a
    COPY of g, vs the baseline (full graph) RMSE. Never mutates g.

    Design choice (regime granularity): "all" — this analysis is NOT split by
    chart. val_samples mix viral50/top200 by construction; splitting per-chart
    would require a second full encode/predict pass per (edge_type, chart) pair
    with no clear interpretability benefit at P2 priority, so a single pooled
    RMSE per edge type is reported (regime="all").
    """
    baseline = _baseline_rmse(model, g, val_samples)
    y_true = np.array([s.y for s in val_samples], dtype=np.float64)

    rows = []
    for et in g.edge_types:
        g_ablated = _empty_edge_store(g, et)
        y_pred = predict_all(model, g_ablated, val_samples)
        rows.append({
            "analysis": "edge_type_ablation",
            "component": str(et),
            "delta_rmse": rmse(y_true, y_pred) - baseline,
            "regime": "all",
        })

    return pd.DataFrame(rows, columns=_SCHEMA)


def feature_group_permutation(
    model: "MusicDiffusionGNN",
    g: HeteroData,
    val_samples: list["Sample"],
    groups: dict[str, list[int]],
) -> pd.DataFrame:
    """Delta RMSE from permuting (shuffling row order across music nodes) one
    group of feature columns on g["music"].x at a time. Never mutates g."""
    baseline = _baseline_rmse(model, g, val_samples)
    y_true = np.array([s.y for s in val_samples], dtype=np.float64)

    n_nodes = g["music"].x.shape[0]
    rng = np.random.default_rng(seed=42)

    rows = []
    for group_name, col_idxs in groups.items():
        g_perturbed = g.clone()
        x = g_perturbed["music"].x
        perm = torch.from_numpy(rng.permutation(n_nodes))
        cols = torch.as_tensor(col_idxs, dtype=torch.long)
        x[:, cols] = x[perm][:, cols]
        g_perturbed["music"].x = x

        y_pred = predict_all(model, g_perturbed, val_samples)
        rows.append({
            "analysis": "feature_group_permutation",
            "component": group_name,
            "delta_rmse": rmse(y_true, y_pred) - baseline,
            "regime": "all",
        })

    return pd.DataFrame(rows, columns=_SCHEMA)


def population_analogs(curves_df: pd.DataFrame) -> pd.DataFrame:
    """SIR-analogous rise/fall descriptive stats fit to GNN-predicted trajectories.

    SPEC_DEVIATION (documented per task spec): this analysis produces beta/gamma/R0
    *values*, not RMSE deltas, but is forced into the same 4-column
    [analysis, component, delta_rmse, regime] schema mandated by design.md §5/R5.4
    for the unified interpretability.parquet table. The `delta_rmse` column is
    therefore reused to carry the actual analog value (beta_analog / gamma_analog /
    r0_analog) for this analysis only — it is NOT an RMSE delta here. Consumers of
    interpretability.parquet must branch on `analysis == "population_analogs"` to
    interpret `delta_rmse` correctly. Column kept as-is (not renamed) per explicit
    instruction in the task spec.
    """
    rows = []
    for (song_id, chart), grp in curves_df.groupby(["song_id", "chart"]):
        grp = grp.sort_values("week")
        y = grp["y_pred_gnn"].to_numpy(dtype=np.float64)
        peak_idx = int(np.argmax(y))

        if peak_idx == 0:
            beta_analog = 0.0
        else:
            rising = y[: peak_idx + 1]
            beta_analog = float(np.mean(np.diff(rising)))

        if peak_idx == len(y) - 1:
            gamma_analog = 0.0
        else:
            falling = y[peak_idx:]
            gamma_analog = float(np.mean(-np.diff(falling)))

        r0_analog = beta_analog / gamma_analog if gamma_analog > 1e-12 else float("nan")

        prefix = f"{song_id}|{chart}"
        rows.append({"analysis": "population_analogs", "component": f"{prefix}|beta", "delta_rmse": beta_analog, "regime": chart})
        rows.append({"analysis": "population_analogs", "component": f"{prefix}|gamma", "delta_rmse": gamma_analog, "regime": chart})
        rows.append({"analysis": "population_analogs", "component": f"{prefix}|r0", "delta_rmse": r0_analog, "regime": chart})

    return pd.DataFrame(rows, columns=_SCHEMA)
