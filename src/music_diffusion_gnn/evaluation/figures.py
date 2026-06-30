"""Phase-3 report figures: Fig.3 boxplot (GNN vs SIR) and Figs.8/9 case curves."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_CHARTS = ["viral50", "top200"]
_CHART_LABELS = {"viral50": "Virality", "top200": "Success"}
_MODELS = ["gnn", "sir"]
_MODEL_LABELS = {"gnn": "GNN", "sir": "SIR"}
_MODEL_COLORS = {"gnn": "#d6604d", "sir": "#4393c3"}


def fig3_boxplot_gnn_vs_sir(mode1_df: pd.DataFrame, out_path: Path) -> Path:
    """Boxplot of per-song RMSE, GNN vs SIR, one subplot per chart regime.

    1x2 grid (viral50, top200), each subplot with two side-by-side boxes
    (GNN, SIR) built from the `rmse` column (full-span RMSE).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(_CHARTS), figsize=(10, 5), squeeze=False)
    axes = axes[0]

    for ax, chart in zip(axes, _CHARTS):
        sub = mode1_df[mode1_df["chart"] == chart]
        data = [sub[sub["model"] == m]["rmse"].dropna().values for m in _MODELS]

        positions = list(range(len(_MODELS)))
        bp = ax.boxplot(
            data, positions=positions, widths=0.5,
            patch_artist=True, showfliers=False,
        )
        for patch, m in zip(bp["boxes"], _MODELS):
            patch.set_facecolor(_MODEL_COLORS[m])
            patch.set_alpha(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels([_MODEL_LABELS[m] for m in _MODELS], fontsize=11)
        ax.set_ylabel("RMSE", fontsize=11)
        ax.set_title(_CHART_LABELS[chart], fontsize=12)
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("Per-song RMSE: GNN vs SIR by chart regime", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figs_8_9_cases(
    curves_df: pd.DataFrame, cases: list[dict], out_path: Path
) -> Path:
    """Multi-panel figure: real vs GNN vs SIR weekly curves for named cases.

    One subplot per case in `cases`. Cases with no matching rows in
    `curves_df` are drawn as an empty panel annotated "no data" rather than
    raising. A case's `note`, if present, is appended to its title.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(cases)
    ncols = 2 if n > 1 else 1
    nrows = max(1, math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    flat_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for ax, case in zip(flat_axes, cases):
        song_id = case["song_id"]
        chart = case["chart"]
        name = case.get("name", song_id)
        note = case.get("note")
        title = f"{name} ({note})" if note else name

        sub = curves_df[
            (curves_df["song_id"] == song_id) & (curves_df["chart"] == chart)
        ].sort_values("week")

        if sub.empty:
            ax.set_title(title, fontsize=11)
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax.plot(sub["week"], sub["y_true"], linestyle="-", color="black", label="Real")
        ax.plot(sub["week"], sub["y_pred_gnn"], linestyle="--", color="#d6604d", label="GNN")
        ax.plot(sub["week"], sub["y_pred_sir"], linestyle=":", color="#4393c3", label="SIR")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("week", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide any unused trailing panels (e.g. odd number of cases in an even grid).
    for ax in flat_axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
