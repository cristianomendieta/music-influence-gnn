"""Training loop, hyperparameter grid, and evaluation for Phase 2."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from music_diffusion_gnn.evaluation.metrics import rmse
from music_diffusion_gnn.models.baselines import persistence_predict
from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
from music_diffusion_gnn.training.dataset import Sample, build_samples

if TYPE_CHECKING:
    from pathlib import Path
    from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    W: int
    hidden: int
    layers: int
    lr: float
    weight_decay: float = 1e-5
    dropout: float = 0.2
    max_epochs: int = 100
    patience: int = 10
    seed: int = 42
    # Week-grouped batching: each sub-batch stays within one target_week so the
    # encoder bank (W forward passes) is computed once per week and reused.
    # Setting batch_size >= max samples per week (≈1756 for 1981 songs, W=4)
    # means each week is processed as a single batch → 1 forward + 1 backward
    # per week instead of n_sub × backward (no retain_graph overhead).
    batch_size: int = 2048
    # Subsample cotrajectory edges per snapshot to this maximum.
    # Necessary when 664K edges exhaust autograd memory on CPU/WSL.
    max_cotraj_edges: int = 30_000

    def __str__(self) -> str:
        return f"W{self.W}_h{self.hidden}_l{self.layers}_lr{self.lr:.0e}"


@dataclass
class TrainResult:
    best_state_dict: dict
    train_curve: list[float]
    val_curve: list[float]
    val_mse: float
    n_params: int
    elapsed_sec: float


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _collate_batch(samples: list[Sample], node_id_map_path=None) -> tuple:
    """Return (y_tensor, samples) — y is (B,) float32."""
    y = torch.tensor([s.y for s in samples], dtype=torch.float32)
    return y, samples


def _iter_batches(
    samples: list[Sample],
    batch_size: int,
    rng: torch.Generator,
) -> list[list[Sample]]:
    """Yield week-grouped batches to minimise distinct encoder calls per batch.

    Groups samples by target_week (so each batch needs at most W window-week
    encoder calls). Week groups are shuffled between epochs via ``rng``.
    Within each group, samples are also shuffled. Groups are concatenated into
    batches of size ``batch_size`` — a batch may straddle at most 2 groups,
    keeping distinct encoder calls at most 2W.
    """
    # Group by target_week
    from collections import defaultdict
    by_week: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_week[s.target_week].append(s)

    # Shuffle week order
    week_keys = list(by_week.keys())
    perm = torch.randperm(len(week_keys), generator=rng).tolist()
    ordered_weeks = [week_keys[i] for i in perm]

    # Build a flat ordered list (shuffled within each week group)
    flat: list[Sample] = []
    for wk in ordered_weeks:
        grp = by_week[wk]
        grp_perm = torch.randperm(len(grp), generator=rng).tolist()
        flat.extend(grp[i] for i in grp_perm)

    # Slice into batches
    batches = []
    for start in range(0, len(flat), batch_size):
        batches.append(flat[start : start + batch_size])
    return batches


def _distinct_window_weeks(samples: list[Sample]) -> list[int]:
    """Collect all distinct (non-padding) week indices needed by the batch."""
    weeks = set()
    for s in samples:
        for wk, is_pad in zip(s.window_weeks, s.pad_mask):
            if not is_pad:
                weeks.add(wk)
    return list(weeks)


# ---------------------------------------------------------------------------
# T9: train_one
# ---------------------------------------------------------------------------

def train_one(
    config: Config,
    splits: dict[str, list[Sample]],
    g,  # HeteroData
    device: str = "cpu",
    pop_bank=None,
) -> TrainResult:
    """Train a single config with early stopping on val MSE.

    Args:
        config: hyperparameters
        splits: dict with 'train', 'val' lists of Sample
        g: the full hetero_full graph (HeteroData)
        device: torch device for model + graph ("cpu" or "cuda")
        pop_bank: optional (n_weeks, N_music, 2) popularity tensor (R1). When
            given, popularity is injected as a node feature and the head is
            anchored to persistence; when None the model is structure-only.

    Returns:
        TrainResult with best state_dict, loss curves, val MSE, params, time
    """
    _set_seed(config.seed)
    t_start = time.time()

    g = g.to(device)
    model = MusicDiffusionGNN(
        g.metadata(),
        n_genre=g["genre"].num_nodes,
        hidden=config.hidden,
        layers=config.layers,
        dropout=config.dropout,
        pop_bank=pop_bank,
    ).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    train_samples = splits["train"]
    val_samples   = splits["val"]

    rng = torch.Generator()
    rng.manual_seed(config.seed)

    best_val_mse = float("inf")
    best_state   = None
    patience_cnt = 0

    train_curve: list[float] = []
    val_curve: list[float]   = []

    # Pre-group train samples by target_week for efficient bank reuse
    from collections import defaultdict
    by_week: dict[int, list[Sample]] = defaultdict(list)
    for s in train_samples:
        by_week[s.target_week].append(s)
    week_keys = list(by_week.keys())

    for epoch in range(config.max_epochs):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        # Shuffle week order each epoch
        week_perm = torch.randperm(len(week_keys), generator=rng).tolist()

        for wi in week_perm:
            w = week_keys[wi]
            week_samples = by_week[w]

            # Window weeks for this target_week (same for all samples in group)
            window_weeks = _distinct_window_weeks(week_samples)
            if not window_weeks:
                continue

            # Compute bank ONCE for the whole week group
            # retain_graph lets us backprop through the same bank for each sub-batch
            bank = model.encode_weeks(g, window_weeks,
                                      max_cotraj_edges=config.max_cotraj_edges)

            # Sub-batch the week group
            grp_perm = torch.randperm(len(week_samples), generator=rng).tolist()
            sub_batches = [
                [week_samples[grp_perm[i]] for i in range(start, min(start + config.batch_size, len(week_samples)))]
                for start in range(0, len(week_samples), config.batch_size)
            ]
            n_sub = len(sub_batches)

            optimizer.zero_grad()
            for bi, batch in enumerate(sub_batches):
                is_last = (bi == n_sub - 1)
                y_hat  = model.predict(bank, batch)
                y_true = torch.tensor([s.y for s in batch], dtype=torch.float32, device=device)
                loss   = loss_fn(y_hat, y_true) / n_sub
                loss.backward(retain_graph=not is_last)
                epoch_loss += loss.item() * n_sub  # undo normalisation for logging
                n_batches  += 1
            optimizer.step()

        train_mse = epoch_loss / max(n_batches, 1)
        train_curve.append(train_mse)

        # ---- Val ----
        val_mse = _eval_mse(model, g, val_samples, config.batch_size,
                            max_cotraj_edges=config.max_cotraj_edges)
        val_curve.append(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config.patience:
                break

    # Reload best weights to count initialized params
    model.load_state_dict(best_state)
    n_params = model.count_params()

    return TrainResult(
        best_state_dict=best_state,
        train_curve=train_curve,
        val_curve=val_curve,
        val_mse=best_val_mse,
        n_params=n_params,
        elapsed_sec=time.time() - t_start,
    )


def _eval_mse(
    model: MusicDiffusionGNN,
    g,
    samples: list[Sample],
    batch_size: int,
    max_cotraj_edges: int | None = None,
) -> float:
    """Compute MSE over samples without updating gradients."""
    model.eval()
    preds, targets = [], []
    rng = torch.Generator()
    rng.manual_seed(0)
    with torch.no_grad():
        for batch in _iter_batches(samples, batch_size, rng):
            weeks = _distinct_window_weeks(batch)
            if not weeks:
                continue
            bank = model.encode_weeks(g, weeks, max_cotraj_edges=max_cotraj_edges)
            y_hat = model.predict(bank, batch)
            preds.append(y_hat.cpu())
            targets.append(torch.tensor([s.y for s in batch], dtype=torch.float32))
    if not preds:
        return float("inf")
    return nn.MSELoss()(torch.cat(preds), torch.cat(targets)).item()


# ---------------------------------------------------------------------------
# T10: run_grid
# ---------------------------------------------------------------------------

# Default Phase 2 grid (3×2×2×2 = 24 configs)
DEFAULT_GRID = [
    Config(W=W, hidden=h, layers=l, lr=lr)
    for W in (4, 8, 12)
    for h in (64, 128)
    for l in (2, 3)
    for lr in (1e-3, 5e-4)
]


def run_grid(
    grid: list[Config],
    splits: dict[str, list[Sample]],
    g,
    device: str = "cpu",
    pop_bank=None,
) -> tuple[pd.DataFrame, Config, dict]:
    """Run hyperparameter grid; return (results_df, best_config, best_state_dict).

    The DataFrame has one row per config with columns:
    config_str, W, hidden, layers, lr, train_mse, val_mse, n_params, elapsed_sec.

    ``pop_bank`` (R1) is forwarded to each :func:`train_one` call.
    """
    rows = []
    best_val_mse = float("inf")
    best_config  = None
    best_state   = None

    for i, cfg in enumerate(grid):
        print(f"  [{i+1}/{len(grid)}] {cfg} ...", flush=True)
        result = train_one(cfg, splits, g, device=device, pop_bank=pop_bank)
        print(f"    val_mse={result.val_mse:.6f}  params={result.n_params}  t={result.elapsed_sec:.1f}s")

        rows.append({
            "config_str":  str(cfg),
            "W":           cfg.W,
            "hidden":      cfg.hidden,
            "layers":      cfg.layers,
            "lr":          cfg.lr,
            "train_mse":   result.train_curve[-1] if result.train_curve else float("nan"),
            "val_mse":     result.val_mse,
            "n_params":    result.n_params,
            "elapsed_sec": result.elapsed_sec,
        })

        if result.val_mse < best_val_mse:
            best_val_mse = result.val_mse
            best_config  = cfg
            best_state   = result.best_state_dict

    df = pd.DataFrame(rows)
    return df, best_config, best_state


# ---------------------------------------------------------------------------
# T11: evaluate
# ---------------------------------------------------------------------------

def evaluate(
    model: MusicDiffusionGNN,
    splits: dict[str, list[Sample]],
    weekly_df: pd.DataFrame,
    val_split_df: pd.DataFrame,
    g,
    mode: str,
    batch_size: int = 64,
    max_cotraj_edges: int | None = None,
    device: str = "cpu",
) -> dict:
    """Evaluate model on val set under the specified protocol.

    Args:
        model: trained MusicDiffusionGNN (best weights loaded)
        splits: dict with 'train', 'val', 'test' Sample lists
        weekly_df: full weekly DataFrame (for persistence baseline lookup)
        val_split_df: weekly_df rows for val split (same songs/weeks as splits['val'])
        g: hetero_full graph
        mode: 'forecasting' (held-out) or 'retroactive' (in-sample teacher-forced)
        batch_size: eval batch size
        device: torch device for model + graph ("cpu" or "cuda")

    Returns:
        dict with keys:
            mse_viral50, mse_top200, rmse_viral50, rmse_top200,
            persist_mse_viral50, persist_mse_top200,
            predictions_df  (columns: song_idx, chart_code, week, y_true, y_pred, mode)
    """
    assert mode in ("forecasting", "retroactive")

    model = model.to(device)
    g = g.to(device)

    if mode == "forecasting":
        eval_samples = splits["val"]
    else:
        # Retroactive: reconstruct full in-sample span (train + val) with teacher-forced windows
        eval_samples = splits["train"] + splits["val"]

    # GNN predictions
    model.eval()
    all_preds, all_true, all_song_idx, all_chart, all_week = [], [], [], [], []
    rng = torch.Generator()
    rng.manual_seed(0)
    with torch.no_grad():
        for batch in _iter_batches(eval_samples, batch_size, rng):
            weeks = _distinct_window_weeks(batch)
            if not weeks:
                continue
            bank = model.encode_weeks(g, weeks, max_cotraj_edges=max_cotraj_edges)
            y_hat = model.predict(bank, batch).cpu().numpy()

            for samp, pred in zip(batch, y_hat):
                all_preds.append(float(pred))
                all_true.append(samp.y)
                all_song_idx.append(samp.song_idx)
                all_chart.append(samp.chart)     # 0=viral50, 1=top200
                all_week.append(samp.target_week)

    preds_arr = np.array(all_preds, dtype=np.float32)
    true_arr  = np.array(all_true, dtype=np.float32)
    chart_arr = np.array(all_chart, dtype=np.int32)

    # Sanity: range check
    assert (preds_arr >= 0).all() and (preds_arr <= 0.5).all(), (
        f"ŷ out of [0,0.5]: min={preds_arr.min()}, max={preds_arr.max()}"
    )

    # Per-regime MSE
    v_mask = chart_arr == 0  # viral50
    s_mask = chart_arr == 1  # top200

    mse_viral50 = float(np.mean((preds_arr[v_mask] - true_arr[v_mask]) ** 2)) if v_mask.any() else float("nan")
    mse_top200  = float(np.mean((preds_arr[s_mask] - true_arr[s_mask]) ** 2)) if s_mask.any() else float("nan")
    rmse_viral50 = float(np.sqrt(mse_viral50)) if not np.isnan(mse_viral50) else float("nan")
    rmse_top200  = float(np.sqrt(mse_top200)) if not np.isnan(mse_top200) else float("nan")

    # Persistence baseline (on the val split regardless of mode, for fair comparison)
    val_df = val_split_df.copy()
    v_val = val_df[val_df["chart"] == "viral50"]
    s_val = val_df[val_df["chart"] == "top200"]

    def _persist_mse(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return float("nan")
        p = persistence_predict(weekly_df, df)
        return float(np.mean((p - df["y_week"].values) ** 2))

    persist_mse_viral50 = _persist_mse(v_val)
    persist_mse_top200  = _persist_mse(s_val)

    # Predictions DataFrame for serialization
    pred_df = pd.DataFrame({
        "song_idx": all_song_idx,
        "chart_code": all_chart,
        "week": all_week,
        "y_true": all_true,
        "y_pred": all_preds,
        "mode": mode,
    })

    return {
        "mse_viral50": mse_viral50,
        "mse_top200": mse_top200,
        "rmse_viral50": rmse_viral50,
        "rmse_top200": rmse_top200,
        "persist_mse_viral50": persist_mse_viral50,
        "persist_mse_top200": persist_mse_top200,
        "predictions_df": pred_df,
    }
