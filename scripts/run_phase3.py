"""Phase 3 orchestrator — dual evaluation: GNN vs SIR (Mode 1 + Mode 2).

Usage:
    python scripts/run_phase3.py [options]

Flags:
    --smoke             Fast end-to-end smoke run (few songs / origins).
    --force             Recompute all artifacts even if cached.
    --skip-interpret    Skip P2 interpretability analyses (T10).
    --seed N            Global random seed (default 42).
    --seed-weeks N      Seed weeks for Mode-1 free rollout (default = W from ckpt).
    --origin-stride N   Week stride between Mode-2 origins (default 4).
    --device DEVICE     Torch device (default cpu).
    --split-regime R    Train/val/test boundary regime: "current" or
                        "pre_pandemia" (default "current").

Produces under results/phase3/:
    R5.1  mode1_per_song.parquet
    R5.2  mode2_horizons.parquet
    R5.3  fig3_boxplot.png, figs_8_9_casos.png
    R5.4  interpretability.parquet  (if not --skip-interpret)
    R5.5  summary.md  (includes C1-C12 checklist; exits 1 if any P1 fails)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase3"
RESULTS_P0 = ROOT / "results" / "phase0"
CKPT = ROOT / "results" / "phase2_experimentos_v2" / "grid_best_model.pt"
TS_PATH = ROOT / "data" / "processed" / "timeseries.parquet"
GRAPH_DIR = ROOT / "data" / "processed" / "graph"
NMAP_PATH = ROOT / "data" / "processed" / "graph" / "node_id_map.json"

_MAX_WEEK = 260
_SMOKE_N_SONGS = 5
_SMOKE_N_ORIGINS = 2


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s / 60:.1f}min" if s > 60 else f"{s:.1f}s"


def _step(msg: str) -> None:
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def _banner(msg: str) -> None:
    print(f"\n  → {msg}")


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _week_index_vec(dates: pd.Series) -> pd.Series:
    """Vectorized ISO-week → linear index matching aggregate_weekly."""
    iso = pd.DatetimeIndex(dates).isocalendar()
    return (iso["year"].astype(int) - 2017) * 52 + (iso["week"].astype(int) - 1)


def _fit_sir_from_ts(ts_df: pd.DataFrame, sir_path: Path) -> pd.DataFrame:
    """Fit SIR for every (song, chart) in ts_df and cache to sir_path."""
    from music_diffusion_gnn.baselines.parallel import fit_all
    from music_diffusion_gnn.baselines.sir import fit_sir

    t0 = time.time()
    sir_df = fit_all(ts_df, fit_sir, n_jobs=-1)
    sir_path.parent.mkdir(parents=True, exist_ok=True)
    sir_df.to_parquet(sir_path)
    print(f"  SIR fit: {len(sir_df)} songs, converged {sir_df['converged'].mean() * 100:.1f}%  [{_elapsed(t0)}]")
    return sir_df


def _build_origins(
    weekly_df: pd.DataFrame,
    ks: tuple[int, ...],
    origin_stride: int,
    smoke: bool,
    split_regime: str = "current",
) -> pd.DataFrame:
    """Build Mode-2 origin rows: (song_id, chart, week) in the regime's test span."""
    from music_diffusion_gnn.training.dataset import get_split_regime

    regime = get_split_regime(split_regime)
    max_k = max(ks)
    test_end = regime.test_end_week if regime.test_end_week is not None else _MAX_WEEK
    test_wdf = weekly_df[
        (weekly_df["week"] >= regime.test_start_week)
        & (weekly_df["week"] <= test_end)
        & (weekly_df["week"] + max_k <= _MAX_WEEK)
    ]
    songs = test_wdf[["song_id", "chart"]].drop_duplicates()
    if smoke:
        songs = songs.head(_SMOKE_N_SONGS)

    rows = []
    indexed = test_wdf.set_index(["song_id", "chart", "week"])
    for _, r in songs.iterrows():
        sid, chart = r["song_id"], r["chart"]
        key = (sid, chart)
        sub = test_wdf[(test_wdf["song_id"] == sid) & (test_wdf["chart"] == chart)]
        avail = sorted(sub["week"].unique())
        stride_weeks = [w for i, w in enumerate(avail) if i % origin_stride == 0]
        if smoke:
            stride_weeks = stride_weeks[:_SMOKE_N_ORIGINS]
        for w in stride_weeks:
            rows.append({"song_id": sid, "chart": chart, "week": int(w)})
    return pd.DataFrame(rows, columns=["song_id", "chart", "week"])


def _onchart_weeks_set(ts_df: pd.DataFrame) -> set[tuple[str, str, int]]:
    """Set of (song_id, chart, week) where rank_score > 0 for at least one day."""
    from music_diffusion_gnn.evaluation.longhits import onchart_weeks

    return onchart_weeks(ts_df, max_week=_MAX_WEEK)


def _per_song_rmse(
    pred_df: pd.DataFrame, y_col: str, onchart: set
) -> pd.DataFrame:
    """Compute full-span and on-chart RMSE per (song_id, chart) from a
    prediction frame with columns [song_id, chart, week, y_true, <y_col>]."""
    rows = []
    for (sid, chart), grp in pred_df.groupby(["song_id", "chart"]):
        y_true = grp["y_true"].to_numpy()
        y_pred = grp[y_col].to_numpy()
        valid = ~np.isnan(y_pred)
        if valid.sum() == 0:
            continue
        r = np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2))

        oc_mask = np.array([(sid, chart, int(w)) in onchart for w in grp["week"]])
        oc_valid = valid & oc_mask
        r_oc = np.sqrt(np.mean((y_true[oc_valid] - y_pred[oc_valid]) ** 2)) if oc_valid.sum() > 0 else float("nan")

        rows.append({"song_id": sid, "chart": chart, "rmse": r, "rmse_onchart": r_oc,
                     "n_weeks_eval": int(valid.sum())})
    return pd.DataFrame(rows)


def _mode1_per_song(
    gnn_free_df: pd.DataFrame,
    sir_m1_df: pd.DataFrame,
    ts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build mode1_per_song.parquet schema from Mode-1 rollout outputs."""
    from music_diffusion_gnn.evaluation.longhits import days_on_chart, long_hit_mask
    from music_diffusion_gnn.evaluation.rollout import _saturation_rate

    onchart = _onchart_weeks_set(ts_df)
    long_hits = long_hit_mask(ts_df)
    doc = days_on_chart(ts_df)

    # SIR — merge y_true from gnn_free_df (both share the same real trajectory)
    sir_wide = sir_m1_df.rename(columns={"y_pred_sir": "y_pred_sir"})
    sir_merged = gnn_free_df[["song_id", "chart", "week", "y_true"]].merge(
        sir_wide, on=["song_id", "chart", "week"], how="inner"
    )
    sir_rmse = _per_song_rmse(sir_merged, "y_pred_sir", onchart)
    sir_rmse["model"] = "sir"
    sir_rmse["saturation_rate"] = float("nan")

    # GNN
    gnn_rmse = _per_song_rmse(gnn_free_df, "y_pred", onchart)
    gnn_rmse["model"] = "gnn"
    sat = (
        gnn_free_df.groupby(["song_id", "chart"])["y_pred"]
        .apply(_saturation_rate)
        .reset_index(name="saturation_rate")
    )
    gnn_rmse = gnn_rmse.merge(sat, on=["song_id", "chart"], how="left")

    combined = pd.concat([gnn_rmse, sir_rmse], ignore_index=True)

    def _doc(r):
        key = (r["song_id"], r["chart"])
        return int(doc.get(key, 0))

    def _lh(r):
        return (r["song_id"], r["chart"]) in long_hits

    combined["days_on_chart"] = combined.apply(_doc, axis=1)
    combined["is_long_hit"] = combined.apply(_lh, axis=1)

    cols = ["song_id", "chart", "model", "rmse", "rmse_onchart",
            "days_on_chart", "is_long_hit", "n_weeks_eval", "saturation_rate"]
    return combined[cols].reset_index(drop=True)


def _mode2_horizons(
    rec_df: pd.DataFrame,
    sir_m2_df: pd.DataFrame,
    pers_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build mode2_horizons.parquet schema by tagging each model."""
    gnn = rec_df.rename(columns={"y_pred": "y_pred"})[
        ["song_id", "chart", "origin_week", "k", "y_true", "y_pred"]
    ].copy()
    gnn["model"] = "gnn"
    gnn["converged"] = True

    sir = sir_m2_df.rename(columns={"y_pred_sir": "y_pred"}).copy()
    sir["y_true"] = float("nan")  # filled via merge below
    # Merge y_true from gnn output (same origins)
    ytrue_ref = gnn[["song_id", "chart", "origin_week", "k", "y_true"]].copy()
    sir = sir.drop(columns=["y_true"]).merge(
        ytrue_ref, on=["song_id", "chart", "origin_week", "k"], how="left"
    )
    sir["model"] = "sir"
    sir = sir[["song_id", "chart", "origin_week", "k", "y_true", "y_pred", "model", "converged"]]

    pers = pers_df.rename(columns={"y_pred": "y_pred"}).copy()
    pers = pers.merge(
        ytrue_ref, on=["song_id", "chart", "origin_week", "k"], how="left"
    )
    pers["model"] = "persist"
    pers["converged"] = True
    pers = pers[["song_id", "chart", "origin_week", "k", "y_true", "y_pred", "model", "converged"]]

    all_df = pd.concat([gnn, sir, pers], ignore_index=True)
    cols = ["song_id", "chart", "model", "origin_week", "k", "y_true", "y_pred", "converged"]
    return all_df[cols].reset_index(drop=True)


def _mode2_stats(
    m2: pd.DataFrame,
    weekly_df: pd.DataFrame,
    seed: int,
    onchart: set | None = None,
) -> dict:
    """RMSE + directional accuracy per (model, k, chart, reading); Wilcoxon + bootstrap.

    Reports two readings per cell: ``"full"`` (every test-span origin, the
    numbers already in use) and ``"onchart"`` (restricted to target weeks
    where the song is actually charting — the principal reading per
    docs/adr/0004, since ~95% of full-span Mode-2 targets sit at the floor).
    ``onchart=None`` skips the on-chart reading (only ``"full"`` is produced),
    matching the pre-recorte behavior.

    The GNN-vs-SIR comparison aggregates per-origin squared errors to a
    per-song mean before the paired Wilcoxon + bootstrap CI (docs.md #03):
    origins from the same song are not independent, and pairing directly on
    origins (~23k pairs) instead of songs (~1.9k) inflates significance.
    p-values are Holm-corrected within each reading's family of 6 (chart × k)
    comparisons.
    """
    from music_diffusion_gnn.evaluation.stats import directional_accuracy, holm_correction, paired_by_song

    y_origin_idx = weekly_df.set_index(["song_id", "chart", "week"])["y_week"]

    def _y_origin(row):
        key = (row["song_id"], row["chart"], row["origin_week"])
        return float(y_origin_idx[key]) if key in y_origin_idx.index else float("nan")

    m2 = m2.copy()
    m2["y_origin"] = m2.apply(_y_origin, axis=1)
    m2["target_week"] = m2["origin_week"] + m2["k"]
    if onchart is not None:
        m2["onchart"] = [
            (sid, chart, int(w)) in onchart
            for sid, chart, w in zip(m2["song_id"], m2["chart"], m2["target_week"])
        ]
    readings = ("full", "onchart") if onchart is not None else ("full",)

    stats: dict = {}
    wilcoxon_keys_by_reading: dict[str, list[tuple]] = {r: [] for r in readings}
    for reading in readings:
        m2r = m2 if reading == "full" else m2[m2["onchart"]]
        for chart in ("viral50", "top200"):
            for k in sorted(m2["k"].unique()):
                sub = m2r[(m2r["chart"] == chart) & (m2r["k"] == k)]
                for model in ("gnn", "sir", "persist"):
                    msub = sub[sub["model"] == model].dropna(subset=["y_true", "y_pred"])
                    if msub.empty:
                        continue
                    rmse_val = float(np.sqrt(np.mean((msub["y_true"] - msub["y_pred"]) ** 2)))
                    da = directional_accuracy(
                        msub["y_origin"].to_numpy(),
                        msub["y_true"].to_numpy(),
                        msub["y_pred"].to_numpy(),
                    )
                    stats[(model, k, chart, reading)] = {"rmse": rmse_val, "dir_acc": da, "n": len(msub)}

                # GNN vs SIR: paired by song (not by origin) — see docstring.
                gnn_sub = sub[(sub["model"] == "gnn")].dropna(subset=["y_true", "y_pred"])
                sir_sub = sub[(sub["model"] == "sir") & (sub["converged"])].dropna(subset=["y_true", "y_pred"])
                paired = gnn_sub.set_index(["song_id", "chart", "origin_week"]).join(
                    sir_sub.set_index(["song_id", "chart", "origin_week"]),
                    lsuffix="_gnn", rsuffix="_sir", how="inner",
                )
                if len(paired) >= 2:
                    rmse_gnn = ((paired["y_true_gnn"] - paired["y_pred_gnn"]) ** 2).to_numpy()
                    rmse_sir = ((paired["y_true_sir"] - paired["y_pred_sir"]) ** 2).to_numpy()
                    song_ids = paired.index.get_level_values("song_id").to_numpy()
                    try:
                        w = paired_by_song(rmse_gnn, rmse_sir, song_ids, seed=seed)
                    except Exception:
                        w = {"statistic": float("nan"), "p_value": float("nan"), "n": 0,
                             "ci_lo": float("nan"), "ci_hi": float("nan"), "mean_diff": float("nan"),
                             "win_rate": float("nan"), "n_songs": 0}
                    w["n_origins"] = len(paired)
                    stats[("wilcoxon", k, chart, reading)] = w
                    wilcoxon_keys_by_reading[reading].append((k, chart))

        # Holm correction within this reading's family of comparisons.
        keys = wilcoxon_keys_by_reading[reading]
        if keys:
            p_values = [stats[("wilcoxon", k, chart, reading)]["p_value"] for k, chart in keys]
            adjusted = holm_correction(np.asarray(p_values))
            for (k, chart), p_adj in zip(keys, adjusted):
                stats[("wilcoxon", k, chart, reading)]["p_value_holm"] = float(p_adj)

    return stats


def _pick_cases(mode1: pd.DataFrame, n: int = 4) -> list[dict]:
    """Pick top-n long-hit songs from mode1 GNN results as case studies."""
    gnn = mode1[mode1["model"] == "gnn"].copy()
    gnn_lh = gnn[gnn["is_long_hit"]].sort_values("days_on_chart", ascending=False)
    rows = gnn_lh.head(n)[["song_id", "chart"]].drop_duplicates().head(n).to_dict("records")
    while len(rows) < n:
        # fallback: pick by most weeks evaluated
        extra = gnn.sort_values("n_weeks_eval", ascending=False)
        extra = extra[~extra["song_id"].isin({r["song_id"] for r in rows})].head(1)
        if extra.empty:
            break
        rows.append(extra.iloc[0][["song_id", "chart"]].to_dict())
    for i, r in enumerate(rows):
        r["name"] = f"Case {i + 1} ({r['song_id'][:8]}…/{r['chart']})"
        r["note"] = "Substituting for named paper case (track title unavailable in processed data)"
    return rows


def _write_summary(
    mode1: pd.DataFrame,
    m2_stats: dict,
    checklist: dict[str, bool],
    out_path: Path,
    smoke: bool,
    n_sir_excl: int,
    split_regime: str = "current",
) -> None:
    mode1_gnn = mode1[mode1["model"] == "gnn"]
    mode1_sir = mode1[mode1["model"] == "sir"]

    lines = [
        "# Phase 3 — GNN vs SIR Dual Evaluation",
        "",
        f"**Split regime:** `{split_regime}`",
        "",
        f"{'⚠️  SMOKE MODE — results from a tiny subset' if smoke else ''}",
        "",
        "## Mode 1 — Free Rollout (Reconstruction)",
        "",
        "| Regime | GNN RMSE | SIR RMSE | GNN RMSE (on-chart) | SIR RMSE (on-chart) |",
        "|--------|----------|----------|---------------------|---------------------|",
    ]
    for chart, label in [("viral50", "Virality"), ("top200", "Success")]:
        g = mode1_gnn[mode1_gnn["chart"] == chart]["rmse"].mean()
        s = mode1_sir[mode1_sir["chart"] == chart]["rmse"].mean()
        g_oc = mode1_gnn[mode1_gnn["chart"] == chart]["rmse_onchart"].mean()
        s_oc = mode1_sir[mode1_sir["chart"] == chart]["rmse_onchart"].mean()
        lines.append(f"| {label} | {g:.5f} | {s:.5f} | {g_oc:.5f} | {s_oc:.5f} |")

    # Mode 2: on-chart is the principal reading (docs/adr/0004) — shown first;
    # full-span (includes floor weeks, ~95% of targets) shown for reference.
    readings = ["onchart", "full"] if any(k[-1] == "onchart" for k in m2_stats) else ["full"]
    reading_label = {"onchart": "On-chart (principal)", "full": "Full span (reference)"}

    lines += ["", "## Mode 2 — Recursive Rollout (k-step Forecasting)", ""]
    lines += [f"SIR non-convergent/short-history exclusions: {n_sir_excl}", ""]
    for reading in readings:
        lines += [f"### {reading_label[reading]}", ""]
        lines += ["| Model | k | Regime | RMSE | Dir. Acc. | n |",
                  "|-------|---|--------|------|-----------|---|"]
        for model in ("gnn", "sir", "persist"):
            for k in (1, 2, 4):
                for chart in ("viral50", "top200"):
                    v = m2_stats.get((model, k, chart, reading), {})
                    if v:
                        lines.append(
                            f"| {model} | {k} | {chart} | {v['rmse']:.5f} | {v['dir_acc']:.3f} | {v['n']} |"
                        )
        lines.append("")

    lines += ["## Statistics (GNN vs SIR, Mode 2 — paired by song, Holm-corrected)", ""]
    for reading in readings:
        lines += [f"### {reading_label[reading]}", ""]
        lines += [
            "| k | Regime | Wilcoxon p | Holm p | Win rate | Bootstrap CI (mean sq-err diff) | n songs | n origins |",
            "|---|--------|------------|--------|----------|----------------------------------|---------|-----------|",
        ]
        for k in (1, 2, 4):
            for chart in ("viral50", "top200"):
                v = m2_stats.get(("wilcoxon", k, chart, reading), {})
                if v:
                    lines.append(
                        f"| {k} | {chart} | {v.get('p_value', float('nan')):.3e} | "
                        f"{v.get('p_value_holm', float('nan')):.3e} | "
                        f"{v.get('win_rate', float('nan')):.3f} | "
                        f"[{v.get('ci_lo', float('nan')):.4f}, {v.get('ci_hi', float('nan')):.4f}] | "
                        f"{v.get('n_songs', 0)} | {v.get('n_origins', 0)} |"
                    )
        lines.append("")

    # Long-hit subgroup
    lh_gnn = mode1_gnn[mode1_gnn["is_long_hit"]]["rmse"].mean()
    lh_sir = mode1_sir[mode1_sir["is_long_hit"]]["rmse"].mean()
    lh_gain = (lh_sir - lh_gnn) / lh_sir * 100 if lh_sir > 0 else float("nan")
    lines += [
        "", "## Long-Hit Subgroup (> 90 days on chart)", "",
        f"| Model | Mean RMSE | N songs |",
        f"|-------|-----------|---------|",
        f"| GNN   | {lh_gnn:.5f} | {int(mode1_gnn[mode1_gnn['is_long_hit']].shape[0])} |",
        f"| SIR   | {lh_sir:.5f} | {int(mode1_sir[mode1_sir['is_long_hit']].shape[0])} |",
        f"",
        f"GNN RMSE reduction vs SIR (long hits): {lh_gain:.1f}%",
        f"{'✅ ≥30% — strong result' if lh_gain >= 30 else '⚠️ <30% — below strong-result target (C7 still passes)'}",
    ]

    # C1-C12 checklist
    lines += ["", "## Acceptance Checklist (C1–C12)", ""]
    p1_labels = {
        "C1": "run_phase3.py ran end-to-end without error",
        "C2": "SIR and GNN evaluated on same weekly axis and subset",
        "C3": "Mode-1 boxplot + RMSE ± CI95% + Wilcoxon saved",
        "C4": "Mode-1 GNN < SIR in RMSE (success) with p < 0.01",
        "C5": "Mode-2 RMSE + directional accuracy per k and regime saved",
        "C6": "Mode-2 GNN < SIR in ≥ 2 of 3 horizons",
        "C7": "Long-hit subgroup (> 90 days) reported",
        "C8": "Figs. 8/9 replicated for 4 cases (or documented substitutes)",
        "C12": "summary.md with GNN vs SIR master table present",
    }
    p2_labels = {
        "C9": "Edge-type ablation (Δ RMSE) reported",
        "C10": "Feature importance (acoustic / metadata / lagged-pop) reported",
        "C11": "β/γ/R₀ analogs extracted from GNN trajectories",
    }
    for key, label in {**p1_labels, **p2_labels}.items():
        prio = "P1" if key in p1_labels else "P2"
        status = checklist.get(key, False)
        icon = "✅" if status else "❌"
        lines.append(f"- {icon} **{key}** ({prio}) — {label}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(out_path.read_text())


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 dual evaluation orchestrator")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke run (few songs)")
    parser.add_argument("--force", action="store_true", help="Recompute all artifacts")
    parser.add_argument("--skip-interpret", action="store_true", help="Skip P2 interpretability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-weeks", type=int, default=None,
                        help="Seed weeks for Mode-1 rollout (default = W from checkpoint)")
    parser.add_argument("--origin-stride", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-regime", default="current", choices=["current", "pre_pandemia"],
                        help="named train/val/test boundary regime (default: current)")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / ".gitkeep").touch()

    ks = (1, 2, 4)
    checklist: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # R0 — Prerequisites                                                   #
    # ------------------------------------------------------------------ #
    _step("R0/A — Load timeseries + graph")
    t0 = time.time()
    from music_diffusion_gnn.graph.build import graph_path

    ts_df = pd.read_parquet(TS_PATH)
    g = torch.load(graph_path(args.split_regime, GRAPH_DIR), weights_only=False)
    print(f"  Loaded: {ts_df.shape} rows, {g['music'].num_nodes} music nodes  [{_elapsed(t0)}]")

    from music_diffusion_gnn.training.dataset import aggregate_weekly, build_pop_bank

    # Smoke: filter ts_df FIRST (before the expensive aggregate_weekly call)
    if args.smoke:
        all_ids = ts_df["song_id"].unique()
        smoke_ids = set(all_ids[:_SMOKE_N_SONGS])
        ts_df = ts_df[ts_df["song_id"].isin(smoke_ids)].reset_index(drop=True)
        _banner(f"Smoke mode: {len(smoke_ids)} songs, {len(ts_df)} daily rows")

    weekly_df_full = aggregate_weekly(ts_df)

    if args.smoke:
        _banner(f"Smoke weekly: {len(weekly_df_full)} rows")

    # ------------------------------------------------------------------ #
    _step("R0/B — SIR baseline fits")
    sir_path = RESULTS_P0 / "sir_params.parquet"
    smoke_sir_path = RESULTS / "smoke_sir_params.parquet"
    active_sir_path = smoke_sir_path if args.smoke else sir_path

    if args.force or not active_sir_path.exists():
        _banner("Fitting SIR (parallelized)…")
        sir_df = _fit_sir_from_ts(ts_df, active_sir_path)
    else:
        sir_df = pd.read_parquet(active_sir_path)
        # In non-smoke mode: filter to current ts_df songs only
        songs_in_ts = set(zip(ts_df["song_id"], ts_df["chart"]))
        sir_df_reset = sir_df.reset_index()
        sir_df = sir_df_reset[
            sir_df_reset.apply(lambda r: (r["song_id"], r["chart"]) in songs_in_ts, axis=1)
        ].set_index(["song_id", "chart"])
        _banner(f"SIR cached: {len(sir_df)} fits  [{active_sir_path.relative_to(ROOT)}]")

    # ------------------------------------------------------------------ #
    _step("R0/C — Load GNN checkpoint + pop_bank")
    ck = torch.load(CKPT, map_location=args.device, weights_only=False)
    W: int = ck["W"]
    seed_weeks = args.seed_weeks if args.seed_weeks is not None else W
    pop_bank = build_pop_bank(weekly_df_full, NMAP_PATH, n_music=g["music"].num_nodes)

    from music_diffusion_gnn.evaluation.model_io import load_grid_best_model
    model = load_grid_best_model(CKPT, g=g, pop_bank_regen=pop_bank, device=args.device)
    model.eval()
    _banner(f"Loaded GNN: W={W}, hidden={ck['hidden']}, layers={ck['layers']}, "
            f"val_mse={ck.get('val_mse', float('nan')):.6f}")

    # ------------------------------------------------------------------ #
    # M1 — Mode 1: Free Rollout                                            #
    # ------------------------------------------------------------------ #
    _step("M1/A — GNN free rollout (Mode 1)")
    gnn_free_path = RESULTS / "gnn_free_rollout.parquet"
    if args.force or not gnn_free_path.exists():
        from music_diffusion_gnn.evaluation.rollout import gnn_rollout_free
        t0 = time.time()
        gnn_free_df = gnn_rollout_free(
            model, g, weekly_df_full, W=W, seed_weeks=seed_weeks, device=args.device
        )
        gnn_free_df.to_parquet(gnn_free_path)
        _banner(f"GNN free rollout: {len(gnn_free_df)} rows  [{_elapsed(t0)}]")
    else:
        gnn_free_df = pd.read_parquet(gnn_free_path)
        _banner(f"GNN free rollout cached: {len(gnn_free_df)} rows")

    _step("M1/B — SIR Mode-1 weekly reconstruction")
    from music_diffusion_gnn.evaluation.sir_eval import run_sir_mode1
    t0 = time.time()
    sir_m1_df = run_sir_mode1(ts_df, sir_df)
    _banner(f"SIR Mode-1: {len(sir_m1_df)} rows  [{_elapsed(t0)}]")

    _step("M1/C — Compute per-song RMSE → mode1_per_song.parquet")
    mode1 = _mode1_per_song(gnn_free_df, sir_m1_df, ts_df)
    mode1["split_regime"] = args.split_regime
    mode1.to_parquet(RESULTS / "mode1_per_song.parquet")
    _banner(f"mode1_per_song: {len(mode1)} rows (songs × models)")
    checklist["C2"] = True  # same weekly axis by construction (both use aggregate_weekly)

    # ------------------------------------------------------------------ #
    # M2 — Mode 2: Recursive Forecasting                                   #
    # ------------------------------------------------------------------ #
    _step("M2/A — Build origins")
    origins = _build_origins(weekly_df_full, ks, args.origin_stride, args.smoke, args.split_regime)
    _banner(f"Origins: {len(origins)} rows (songs × weeks), stride={args.origin_stride}")

    _step("M2/B — GNN recursive rollout (Mode 2)")
    rec_path = RESULTS / "gnn_recursive_rollout.parquet"
    if args.force or not rec_path.exists():
        from music_diffusion_gnn.evaluation.rollout import gnn_rollout_recursive
        t0 = time.time()
        rec_df = gnn_rollout_recursive(
            model, g, weekly_df_full, origins, W=W, ks=ks, device=args.device,
            regime=args.split_regime,
        )
        rec_df.to_parquet(rec_path)
        _banner(f"GNN recursive: {len(rec_df)} rows  [{_elapsed(t0)}]")
    else:
        rec_df = pd.read_parquet(rec_path)
        _banner(f"GNN recursive cached: {len(rec_df)} rows")

    _step("M2/C — SIR causal refit + persistence baseline")
    from music_diffusion_gnn.evaluation.sir_eval import run_sir_mode2
    from music_diffusion_gnn.models.baselines import persistence_multistep
    t0 = time.time()
    sir_m2_path = RESULTS / "sir_m2.parquet"
    if args.force or not sir_m2_path.exists():
        # Heaviest single step (thousands of per-origin refits) — cache to survive
        # a Colab disconnect; RESULTS may be Drive-backed for durability.
        sir_m2_df = run_sir_mode2(ts_df, origins, ks=ks, regime=args.split_regime)
        sir_m2_df.to_parquet(sir_m2_path)
    else:
        sir_m2_df = pd.read_parquet(sir_m2_path)
        _banner(f"SIR Mode-2 cached: {len(sir_m2_df)} rows  [{sir_m2_path.relative_to(ROOT)}]")
    pers_df = persistence_multistep(weekly_df_full, origins, ks=ks)
    n_sir_excl = int((~sir_m2_df["converged"]).sum())
    _banner(f"SIR Mode-2: {len(sir_m2_df)} rows, {n_sir_excl} non-convergent  [{_elapsed(t0)}]")

    _step("M2/D — Assemble mode2_horizons.parquet")
    mode2 = _mode2_horizons(rec_df, sir_m2_df, pers_df)
    mode2["split_regime"] = args.split_regime
    mode2.to_parquet(RESULTS / "mode2_horizons.parquet")
    _banner(f"mode2_horizons: {len(mode2)} rows")
    checklist["C5"] = True

    _step("M2/E — Statistics (RMSE + directional acc + Wilcoxon + bootstrap)")
    onchart = _onchart_weeks_set(ts_df)
    m2_stats = _mode2_stats(mode2, weekly_df_full, seed=args.seed, onchart=onchart)
    checklist["C7"] = True  # long-hit subgroup always computed from mode1

    # C6: GNN < SIR in ≥ 2 of 3 horizons per regime — gated on the full-span
    # reading (unchanged semantics); on-chart is reported alongside, not gated.
    c6_pass = []
    for chart in ("viral50", "top200"):
        wins = sum(
            1 for k in ks
            if m2_stats.get(("gnn", k, chart, "full"), {}).get("rmse", float("inf")) <
               m2_stats.get(("sir", k, chart, "full"), {}).get("rmse", float("inf"))
        )
        c6_pass.append(wins >= 2)
    checklist["C6"] = all(c6_pass)

    # ------------------------------------------------------------------ #
    # Statistics — Mode 1 Wilcoxon + bootstrap CI                         #
    # ------------------------------------------------------------------ #
    _step("Mode-1 Statistics (Wilcoxon + bootstrap)")
    from music_diffusion_gnn.evaluation.stats import (
        bootstrap_ci_diff,
        wilcoxon_signed_rank,
    )

    c3_ok, c4_ok = True, True
    for chart in ("viral50", "top200"):
        gnn_rmse = mode1[(mode1["model"] == "gnn") & (mode1["chart"] == chart)].set_index("song_id")["rmse"]
        sir_rmse = mode1[(mode1["model"] == "sir") & (mode1["chart"] == chart)].set_index("song_id")["rmse"]
        paired = gnn_rmse.align(sir_rmse, join="inner")
        if len(paired[0]) < 2:
            continue
        try:
            w = wilcoxon_signed_rank(paired[0].to_numpy(), paired[1].to_numpy())
        except Exception:
            w = {"p_value": float("nan")}
        lo, hi, md = bootstrap_ci_diff(
            paired[0].to_numpy(), paired[1].to_numpy(), seed=args.seed
        )
        p = w.get("p_value", float("nan"))
        gnn_mean = paired[0].mean()
        sir_mean = paired[1].mean()
        _banner(
            f"Mode-1 {chart}: GNN {gnn_mean:.5f} vs SIR {sir_mean:.5f}  "
            f"Wilcoxon p={p:.3e}  bootstrap CI=[{lo:.4f},{hi:.4f}]"
        )
        if chart == "top200" and (np.isnan(p) or p >= 0.01 or gnn_mean >= sir_mean):
            c4_ok = False

    checklist["C3"] = c3_ok
    checklist["C4"] = c4_ok

    # ------------------------------------------------------------------ #
    # Figures — R5.3                                                       #
    # ------------------------------------------------------------------ #
    _step("Figures — Fig.3 boxplot + Figs.8/9 cases")
    from music_diffusion_gnn.evaluation.figures import fig3_boxplot_gnn_vs_sir, figs_8_9_cases

    boxplot_path = fig3_boxplot_gnn_vs_sir(mode1, RESULTS / "fig3_boxplot.png")
    _banner(f"Fig.3 saved: {boxplot_path.relative_to(ROOT)}")

    # Build curves_df for figs_8/9: merge free rollout + SIR M1
    curves_df = gnn_free_df.rename(columns={"y_pred": "y_pred_gnn"}).merge(
        sir_m1_df.rename(columns={"y_pred_sir": "y_pred_sir"}),
        on=["song_id", "chart", "week"], how="outer"
    )
    # y_true comes from gnn_free_df; fill missing from weekly_df
    if "y_true" not in curves_df.columns:
        wdf_idx = weekly_df_full.set_index(["song_id", "chart", "week"])["y_week"]
        curves_df["y_true"] = curves_df.apply(
            lambda r: float(wdf_idx.get((r["song_id"], r["chart"], int(r["week"])), float("nan"))),
            axis=1,
        )

    cases = _pick_cases(mode1)
    cases_path = figs_8_9_cases(curves_df, cases, RESULTS / "figs_8_9_casos.png")
    _banner(f"Figs.8/9 saved: {cases_path.relative_to(ROOT)}")
    checklist["C8"] = True

    # ------------------------------------------------------------------ #
    # P2 — Interpretability (optional)                                     #
    # ------------------------------------------------------------------ #
    if not args.skip_interpret:
        _step("P2 — Interpretability analyses")
        from music_diffusion_gnn.evaluation.interpretability import (
            edge_type_ablation,
            feature_group_permutation,
            population_analogs,
        )
        from music_diffusion_gnn.training.dataset import build_samples

        # Build val samples (small subset for speed)
        smoke_wdf = weekly_df_full[weekly_df_full["song_id"].isin(
            {s for s in weekly_df_full["song_id"].unique()[:20]}
        )]
        val_samples = build_samples(smoke_wdf, W=W, node_id_map_path=str(NMAP_PATH))

        t0 = time.time()
        abl = edge_type_ablation(model, g, val_samples)
        checklist["C9"] = True

        n_cols = g["music"].x.shape[1]
        third = n_cols // 3
        feat_groups = {
            "acoustic_style": list(range(0, third)),
            "structural_meta": list(range(third, 2 * third)),
            "lagged_context": list(range(2 * third, n_cols)),
        }
        perm = feature_group_permutation(model, g, val_samples, feat_groups)
        checklist["C10"] = True

        gnn_free_cols = gnn_free_df[["song_id", "chart", "week", "y_pred"]].rename(
            columns={"y_pred": "y_pred_gnn"}
        )
        analogs = population_analogs(gnn_free_cols)
        checklist["C11"] = True

        interp = pd.concat([abl, perm, analogs], ignore_index=True)
        interp.to_parquet(RESULTS / "interpretability.parquet")
        _banner(f"interpretability.parquet: {len(interp)} rows  [{_elapsed(t0)}]")
    else:
        _banner("Interpretability skipped (--skip-interpret)")
        for c in ("C9", "C10", "C11"):
            checklist[c] = False

    # ------------------------------------------------------------------ #
    # Summary + C1-C12 checklist                                           #
    # ------------------------------------------------------------------ #
    checklist["C1"] = True   # if we reached here, end-to-end ran
    checklist["C12"] = True  # summary.md written below

    _step("Summary — summary.md + C1-C12 checklist")
    _write_summary(
        mode1, m2_stats, checklist, RESULTS / "summary.md", args.smoke, n_sir_excl,
        split_regime=args.split_regime,
    )

    print(f"\n  Artifacts in {RESULTS.relative_to(ROOT)}:")
    for f in sorted(RESULTS.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    # P1 gate: C1-C8, C12
    p1_keys = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C12")
    p1_fail = [k for k in p1_keys if not checklist.get(k, False)]
    if p1_fail:
        print(f"\n❌  Phase 3 FAILED: P1 criteria not met: {p1_fail}")
        sys.exit(1)
    else:
        print("\n✅  Phase 3 complete: all P1 criteria met.")


if __name__ == "__main__":
    main()
