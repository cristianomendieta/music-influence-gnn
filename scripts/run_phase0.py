"""Phase 0 orchestrator — idempotent pipeline end-to-end.

Usage:
    python scripts/run_phase0.py [--force]

--force : re-run all steps even if artifacts exist
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "phase0"


def _elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s/60:.1f}min" if s > 60 else f"{s:.1f}s"


def _step(name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def main(force: bool = False) -> None:
    import pandas as pd

    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1 — Subset                                                      #
    # ------------------------------------------------------------------ #
    subset_path = PROCESSED / "subset_ids.json"
    if force or not subset_path.exists():
        _step("Step 1/4 — Build subset viral∩hit")
        from music_diffusion_gnn.data.subset import build_subset
        t0 = time.time()
        ids = build_subset(subset_path)
        print(f"  Subset size: {len(ids)}  [{_elapsed(t0)}]")
    else:
        from music_diffusion_gnn.data.subset import load_subset
        ids = load_subset(subset_path)
        print(f"Step 1/4 — subset cached ({len(ids)} songs).")

    # ------------------------------------------------------------------ #
    # Step 2 — Timeseries parquet                                          #
    # ------------------------------------------------------------------ #
    ts_path = PROCESSED / "timeseries.parquet"
    if force or not ts_path.exists():
        _step("Step 2/4 — Build timeseries parquet")
        from music_diffusion_gnn.data.preprocess import build_timeseries
        t0 = time.time()
        ts = build_timeseries(out_path=ts_path, subset_ids=ids)
        print(f"  Shape: {ts.shape}  [{_elapsed(t0)}]")
    else:
        from music_diffusion_gnn.data.preprocess import load_timeseries
        ts = load_timeseries(ts_path)
        print(f"Step 2/4 — timeseries cached ({ts.shape}).")

    # ------------------------------------------------------------------ #
    # Step 3 — SIR baseline                                                #
    # ------------------------------------------------------------------ #
    sir_path = RESULTS / "sir_params.parquet"
    if force or not sir_path.exists():
        _step("Step 3/4 — SIR fit (parallelized)")
        from music_diffusion_gnn.baselines.parallel import fit_all
        from music_diffusion_gnn.baselines.sir import fit_sir
        t0 = time.time()
        sir_df = fit_all(ts, fit_sir, n_jobs=-1)
        sir_df.to_parquet(sir_path)
        print(f"  Converged: {sir_df.converged.mean()*100:.1f}%  [{_elapsed(t0)}]")
    else:
        sir_df = pd.read_parquet(sir_path)
        print(f"Step 3/4 — SIR cached ({sir_df.shape}).")

    # ------------------------------------------------------------------ #
    # Step 4 — Report + boxplot                                            #
    # ------------------------------------------------------------------ #
    _step("Step 4/4 — Evaluation and report")
    from music_diffusion_gnn.evaluation.report import make_boxplot, write_summary

    boxplot_path = RESULTS / "boxplot_fig3.png"
    sir_rmse_df = sir_df[["rmse"]].rename(columns={"rmse": "rmse_sir"}).reset_index()
    sir_rmse_df = sir_rmse_df.set_index(["song_id", "chart"])

    make_boxplot(sir_rmse_df, out_path=boxplot_path)
    print(f"  Boxplot saved: {boxplot_path.relative_to(ROOT)}")

    summary_path = write_summary(sir_params_path=sir_path, out_path=RESULTS / "summary.md")
    print(f"  Summary saved: {summary_path.relative_to(ROOT)}")

    print()
    print(summary_path.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-run all steps")
    args = parser.parse_args()
    main(force=args.force)
