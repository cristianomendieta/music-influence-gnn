"""Phase 2 — runner headless e RESILIENTE (espelha o notebook phase2_pipeline_treino).

Diferenças vs. scripts/run_phase2.py:
  - first_seen GLOBAL (corrige super-mascaramento de val/test)
  - avalia val (forecasting + retroactive) E test held-out (semana >= 208)
  - grid com CHECKPOINT INCREMENTAL + RETOMADA (sobrevive a qualquer corte)
  - escreve em results/phase2_experimentos/ (NÃO toca em results/phase2/)

Uso:
    python scripts/run_phase2_experimentos.py --mode single   # 1 config, rápido
    python scripts/run_phase2_experimentos.py --mode grid     # 24 configs, ~2h CPU
    python scripts/run_phase2_experimentos.py --mode grid --smoke   # validação rápida
    python scripts/run_phase2_experimentos.py --mode grid --no-resume

Retomada: rode o mesmo comando de novo — configs já no grid_results.parquet são puladas.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GRAPH_DIR = ROOT / "data" / "processed" / "graph"
NMAP_PATH = GRAPH_DIR / "node_id_map.json"
RESULTS = ROOT / "results" / "phase2_experimentos"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(mode: str, smoke: bool, resume: bool, seed: int) -> int:
    import numpy as np
    import pandas as pd
    import torch

    from music_diffusion_gnn.training.dataset import (
        aggregate_weekly, temporal_split, build_samples,
        TRAIN_END_WEEK, TEST_START_WEEK,
    )
    from music_diffusion_gnn.training.trainer import (
        Config, train_one, evaluate, DEFAULT_GRID,
    )
    from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN

    torch.manual_seed(seed)
    np.random.seed(seed)
    RESULTS.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={DEVICE}  mode={mode}  smoke={smoke}  resume={resume}")

    # ---- dados ----
    g = torch.load(GRAPH_DIR / "hetero_full.pt", weights_only=False).to(DEVICE)
    ts = pd.read_parquet(ROOT / "data" / "processed" / "timeseries.parquet")
    weekly = aggregate_weekly(ts)
    splits_df = temporal_split(weekly)
    log(f"grafo {g.num_nodes} nós | train/val/test rows = "
        f"{len(splits_df['train'])}/{len(splits_df['val'])}/{len(splits_df['test'])}")
    log(f"fronteiras: train<= {TRAIN_END_WEEK}  val..{TEST_START_WEEK-1}  test>= {TEST_START_WEEK}")

    # ---- first_seen GLOBAL ----
    fs_global = weekly.groupby(["song_id", "chart"])["week"].min().to_dict()

    if smoke:
        common = (set(splits_df["train"]["song_id"]) &
                  set(splits_df["val"]["song_id"]) &
                  set(splits_df["test"]["song_id"]))
        song_filter = set(sorted(common)[:40])
        log(f"SMOKE: {len(song_filter)} músicas")
    else:
        song_filter = None

    def make_samples(W):
        out = []
        for name in ("train", "val", "test"):
            df = splits_df[name]
            if song_filter is not None:
                df = df[df["song_id"].isin(song_filter)]
            out.append(build_samples(df, W=W, node_id_map_path=NMAP_PATH, first_seen=fs_global))
        return out  # [train, val, test]

    # ---- grid de configs ----
    if mode == "single":
        grid = [Config(W=4, hidden=64, layers=2, lr=1e-3,
                       max_epochs=5 if smoke else 100,
                       patience=3 if smoke else 10, seed=seed)]
    else:
        grid = [dataclasses.replace(c, seed=seed,
                                    max_epochs=5 if smoke else c.max_epochs,
                                    patience=3 if smoke else c.patience)
                for c in DEFAULT_GRID]
    log(f"{len(grid)} config(s) a treinar")

    GRID_PARQUET = RESULTS / "grid_results.parquet"
    GRID_BEST = RESULTS / "grid_best_model.pt"

    # samples por W (cada Config tem seu W)
    samples_by_W = {}
    for W in sorted({c.W for c in grid}):
        t_, v_, te_ = make_samples(W)
        samples_by_W[W] = {"train": t_, "val": v_, "test": te_}
        log(f"  samples W={W}: train={len(t_)} val={len(v_)} test={len(te_)}")

    # retomada
    rows, done = [], set()
    if resume and GRID_PARQUET.exists():
        rows = pd.read_parquet(GRID_PARQUET).to_dict("records")
        done = {r["config_str"] for r in rows}
        log(f"RETOMADA: {len(done)} config(s) já feitas — serão puladas.")

    best = {"val_mse": float("inf")}
    for r in rows:
        if r["val_mse"] < best["val_mse"]:
            best = {"val_mse": r["val_mse"], "config_str": r["config_str"]}

    t_all = time.time()
    for i, c in enumerate(grid):
        tag = str(c)
        if tag in done:
            log(f"  [{i+1}/{len(grid)}] {tag} — já feito, pulando.")
            continue
        log(f"  [{i+1}/{len(grid)}] {tag} ...")
        s = samples_by_W[c.W]
        r = train_one(c, {"train": s["train"], "val": s["val"]}, g, device=DEVICE)
        log(f"      val_mse={r.val_mse:.6f}  params={r.n_params}  t={r.elapsed_sec:.1f}s")
        rows.append({"config_str": tag, "W": c.W, "hidden": c.hidden, "layers": c.layers,
                     "lr": c.lr, "val_mse": r.val_mse,
                     "train_mse": r.train_curve[-1] if r.train_curve else float("nan"),
                     "n_params": r.n_params, "elapsed_sec": r.elapsed_sec})

        # checkpoint incremental: grava após CADA config
        pd.DataFrame(rows).sort_values("val_mse").reset_index(drop=True).to_parquet(GRID_PARQUET, index=False)
        if r.val_mse < best["val_mse"]:
            best = {"val_mse": r.val_mse, "cfg": c, "state": r.best_state_dict}
            torch.save({"config_str": tag, "W": c.W, "hidden": c.hidden, "layers": c.layers,
                        "lr": c.lr, "dropout": c.dropout, "val_mse": r.val_mse,
                        "state_dict": {k: v.cpu() for k, v in r.best_state_dict.items()}}, GRID_BEST)
            log(f"      ^ novo melhor (val_mse={r.val_mse:.6f}) salvo em {GRID_BEST.name}")

    # reconstrói best do disco (garante avaliação mesmo após retomada)
    ck = torch.load(GRID_BEST, map_location="cpu", weights_only=False)
    best_cfg = Config(W=ck["W"], hidden=ck["hidden"], layers=ck["layers"],
                      lr=ck["lr"], dropout=ck["dropout"])
    best_state = ck["state_dict"]
    log(f"grid concluída em {(time.time()-t_all)/60:.1f}min | melhor={best_cfg} val_mse={ck['val_mse']:.6f}")

    grid_df = pd.DataFrame(rows).sort_values("val_mse").reset_index(drop=True)

    # ---- avaliação do melhor: val (fc + retro) + test held-out ----
    log("avaliando melhor modelo (val forecasting + retroactive + test held-out)...")
    s = samples_by_W[best_cfg.W]
    tr, va, te = s["train"], s["val"], s["test"]
    model = MusicDiffusionGNN(g.metadata(), n_genre=g["genre"].num_nodes,
                              hidden=best_cfg.hidden, layers=best_cfg.layers,
                              dropout=best_cfg.dropout)
    model.load_state_dict(best_state)
    model.to(DEVICE).eval()

    fc_val = evaluate(model=model, splits={"train": tr, "val": va}, weekly_df=weekly,
                      val_split_df=splits_df["val"], g=g, mode="forecasting",
                      max_cotraj_edges=None, device=DEVICE)
    retro_val = evaluate(model=model, splits={"train": tr, "val": va}, weekly_df=weekly,
                         val_split_df=splits_df["val"], g=g, mode="retroactive",
                         max_cotraj_edges=None, device=DEVICE)
    fc_test = evaluate(model=model, splits={"train": tr, "val": te}, weekly_df=weekly,
                       val_split_df=splits_df["test"], g=g, mode="forecasting",
                       max_cotraj_edges=None, device=DEVICE)
    fc_test["predictions_df"]["mode"] = "test"

    # ---- artefatos ----
    preds_all = pd.concat([fc_val["predictions_df"], retro_val["predictions_df"],
                           fc_test["predictions_df"]], ignore_index=True)
    preds_all.to_parquet(RESULTS / "predictions.parquet", index=False)
    torch.save({k: v.cpu() for k, v in best_state.items()}, RESULTS / "best_model.pt")
    grid_df.to_parquet(GRID_PARQUET, index=False)

    def better(a, b):
        return "✓ melhor" if a < b else "✗ pior"

    lines = [
        "# Phase 2 — Experimentos (runner headless)\n\n",
        f"**Melhor config:** `{best_cfg}`  |  **val_mse(grid)** = {ck['val_mse']:.6f}\n",
        f"**Configs na grid:** {len(grid_df)}  |  **SMOKE:** {smoke}\n\n",
        "## Forecasting — VAL (held-out)\n\n",
        "| Regime | GNN MSE | GNN RMSE | Persistência MSE | Δ |\n",
        "|--------|---------|----------|------------------|---|\n",
        f"| viral50 | {fc_val['mse_viral50']:.6f} | {fc_val['rmse_viral50']:.6f} | {fc_val['persist_mse_viral50']:.6f} | {better(fc_val['mse_viral50'], fc_val['persist_mse_viral50'])} |\n",
        f"| top200  | {fc_val['mse_top200']:.6f} | {fc_val['rmse_top200']:.6f} | {fc_val['persist_mse_top200']:.6f} | {better(fc_val['mse_top200'], fc_val['persist_mse_top200'])} |\n\n",
        "## Forecasting — TEST (held-out, semana >= 208)\n\n",
        "| Regime | GNN MSE | GNN RMSE | Persistência MSE | Δ |\n",
        "|--------|---------|----------|------------------|---|\n",
        f"| viral50 | {fc_test['mse_viral50']:.6f} | {fc_test['rmse_viral50']:.6f} | {fc_test['persist_mse_viral50']:.6f} | {better(fc_test['mse_viral50'], fc_test['persist_mse_viral50'])} |\n",
        f"| top200  | {fc_test['mse_top200']:.6f} | {fc_test['rmse_top200']:.6f} | {fc_test['persist_mse_top200']:.6f} | {better(fc_test['mse_top200'], fc_test['persist_mse_top200'])} |\n\n",
        "## Retroactive — VAL (in-sample train+val)\n\n",
        f"- viral50 MSE={retro_val['mse_viral50']:.6f}  RMSE={retro_val['rmse_viral50']:.6f}\n",
        f"- top200  MSE={retro_val['mse_top200']:.6f}  RMSE={retro_val['rmse_top200']:.6f}\n\n",
        "## Artefatos\n\n",
        "- `results/phase2_experimentos/best_model.pt`\n",
        "- `results/phase2_experimentos/grid_results.parquet`\n",
        "- `results/phase2_experimentos/predictions.parquet`\n",
    ]
    (RESULTS / "summary.md").write_text("".join(lines))

    log("DONE. Artefatos:")
    for p in sorted(RESULTS.iterdir()):
        log(f"   {p.name}")
    print("\n" + (RESULTS / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "grid"], default="grid")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    sys.exit(main(args.mode, args.smoke, args.resume, args.seed))
