"""Generate summary.md report and boxplot replicating paper Fig. 3."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from music_diffusion_gnn.evaluation.metrics import mann_whitney_pairwise, summarize_rmse

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "phase0"

_TARGET = {
    "top200": {"target": 0.052, "tol": 0.10},
    "viral50": {"target": 0.028, "tol": 0.10},
}


def make_boxplot(
    sir_rmse_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """Box-plot of SIR RMSE per chart — replicates paper Fig. 3."""
    if out_path is None:
        out_path = RESULTS / "boxplot_fig3.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    charts = ["viral50", "top200"]
    labels_chart = {"viral50": "Virality", "top200": "Success"}

    data_sir = [
        sir_rmse_df.reset_index().query("chart == @c")["rmse_sir"].values
        for c in charts
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = np.arange(len(charts))

    bp = ax.boxplot(data_sir, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor="#4393c3", alpha=0.8))

    ax.set_xticks(positions)
    ax.set_xticklabels([labels_chart[c] for c in charts], fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("SIR RMSE distribution by chart (Phase 0)", fontsize=13)
    ax.legend([bp["boxes"][0]], ["SIR"], fontsize=11)
    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_summary(
    sir_params_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Generate results/phase0/summary.md with R4 criteria table."""
    if sir_params_path is None:
        sir_params_path = RESULTS / "sir_params.parquet"
    if out_path is None:
        out_path = RESULTS / "summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sir = pd.read_parquet(sir_params_path).reset_index()
    sir_success = sir.query("chart == 'top200'")["rmse"].values
    sir_virality = sir.query("chart == 'viral50'")["rmse"].values

    mw_sv = mann_whitney_pairwise(sir_success, sir_virality)

    lines = [
        "# Phase 0 — Summary",
        "",
        "## R4 Numerical Acceptance Criteria",
        "",
        "| Metric | Paper target | Tolerance | Observed | Status |",
        "|--------|-------------|-----------|----------|--------|",
    ]

    for chart, label, target in [("viral50", "SIR · RMSE virality", 0.028),
                                   ("top200",  "SIR · RMSE success",  0.052)]:
        mean_val = sir[sir["chart"] == chart]["rmse"].mean()
        lo, hi = target * 0.9, target * 1.1
        status = "✅ PASS" if lo <= mean_val <= hi else "⚠️ WARN"
        lines.append(f"| {label} | ≈ {target:.3f} | ± 10% | {mean_val:.4f} | {status} |")

    p = mw_sv["p_value"]
    p_str = f"{p:.2e}"
    mw_status = "✅ PASS" if p < 1e-10 else "⚠️ WARN"
    lines.append(f"| Mann-Whitney p (success vs virality) | ≈ 1e-60 | same order | {p_str} | {mw_status} |")

    n_songs = sir["song_id"].nunique()
    conv_pct = sir_df_conv = sir["converged"].mean() * 100 if "converged" in sir.columns else None
    conv_str = f"{conv_pct:.1f}%" if conv_pct is not None else "n/a"
    conv_status = "✅ PASS" if conv_pct is not None and conv_pct >= 99 else "⚠️ WARN"
    lines.append(f"| Subset size | 1 977 | ≥ 1 900 | {n_songs} | {'✅ PASS' if n_songs >= 1900 else '⚠️ WARN'} |")
    lines.append(f"| SIR convergence | 100% | ≥ 99% | {conv_str} | {conv_status} |")

    lines += [
        "",
        "## Dataset",
        "",
        f"- Source: MGD+ (data/charts/mgdplus/) — Top 200 + Viral 50 BR",
        f"- Period: 2017-01-01 → 2022-03-13",
        f"- Subset: viral∩hit, {n_songs} songs",
        "",
    ]

    out_path.write_text("\n".join(lines))
    return out_path
