"""GNN rollout for Phase 3 evaluation — free (Mode 1) and recursive (Mode 2).

Mode 1 resolves OQ1 (fairness of the GNN-vs-SIR comparison): under plain
teacher forcing, ``predict`` reads ``y_prev = pop_bank[w-1]`` = the REAL
previous value, so the residual head degenerates to near-persistence and
beats the SIR baseline trivially (the SIR never sees ``y(w-1)``). The free
rollout instead seeds each (song, chart) with its first ``seed_weeks`` real
weeks, then reconstructs the rest of the trajectory by feeding the model's
own predictions back into the popularity bank — both the injected node
feature ``pop_bank[w]`` and the persistence anchor ``y_prev=pop_bank[w-1]``
become predicted values outside the seed. This is the direct analogue of the
SIR's fit-then-simulate evaluation.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import torch

from music_diffusion_gnn.training.dataset import Sample

_CHART_CODE = {"viral50": 0, "top200": 1}


def gnn_rollout_free(
    model,
    g,
    weekly_df: pd.DataFrame,
    *,
    W: int,
    seed_weeks: int,
    device: str = "cpu",
) -> pd.DataFrame:
    """Free, globally-synchronized GNN rollout (Mode 1, OQ1).

    For each ``(song_id, chart)``, the first ``seed_weeks`` real weeks anchor
    the reconstruction; every later week's prediction is written back into a
    working copy of ``model.pop_bank`` before moving to the next week, so
    later weeks' lookback windows and persistence anchors see predicted (not
    real) values. Processed week-by-week in increasing order, all active
    songs at once, with each distinct encoder week computed at most once
    (lazy cache) — one global pass over the graph (S6), not
    O(songs × span × W).

    Args:
        model: a ``MusicDiffusionGNN`` with a non-None ``pop_bank`` buffer.
        g: the full ``hetero_full`` graph.
        weekly_df: output of ``aggregate_weekly`` — ``[song_id, chart, week,
            y_week]`` — restricted to whatever subset of songs/weeks should
            be rolled out.
        W: look-back window length (must match the model's trained config).
        seed_weeks: number of real weeks used to seed each trajectory before
            switching to self-fed predictions. Edge case: ``span <= W`` =>
            ``seed = max(1, span - 1)`` (logged); ``span < 2`` => the
            (song, chart) is excluded entirely (logged).
        device: torch device for model + graph.

    Returns:
        DataFrame with columns ``[song_id, chart, week, y_true, y_pred]``,
        one row per evaluated ``(song, chart, week)`` — i.e. weeks in
        ``[first_seen + seed .. last_seen]``, excluding the seed window
        itself (which is not a "prediction").
    """
    model = model.to(device)
    g = g.to(device)
    model.eval()
    assert model.pop_bank is not None, "gnn_rollout_free requires a model with a pop_bank buffer"

    song_ids = g["music"].song_id
    song_to_idx = {sid: i for i, sid in enumerate(song_ids)}

    grp = weekly_df.groupby(["song_id", "chart"])["week"]
    first_seen = grp.min().to_dict()
    last_seen = grp.max().to_dict()
    weekly_indexed = weekly_df.set_index(["song_id", "chart", "week"])["y_week"]

    eval_start: dict[tuple[str, str], int] = {}
    n_excluded = 0
    n_seed_clamped = 0
    for key, fsw in first_seen.items():
        lsw = last_seen[key]
        span = lsw - fsw + 1
        if span < 2:
            n_excluded += 1
            continue
        seed = seed_weeks if span > seed_weeks else max(1, span - 1)
        if seed != seed_weeks:
            n_seed_clamped += 1
        eval_start[key] = fsw + seed

    print(
        f"gnn_rollout_free: {len(eval_start)} (song,chart) rolled out, "
        f"{n_excluded} excluded (span<2), {n_seed_clamped} seed-clamped (span<=W)."
    )

    if not eval_start:
        return pd.DataFrame(columns=["song_id", "chart", "week", "y_true", "y_pred"])

    orig_bank = model.pop_bank
    work_bank = orig_bank.clone()
    model.pop_bank = work_bank

    max_week = max(last_seen[key] for key in eval_start)
    zcache: dict[int, torch.Tensor] = {}
    rows: list[dict] = []

    try:
        for w in range(1, max_week + 1):
            active_keys = [
                key for key, start in eval_start.items() if start <= w <= last_seen[key]
            ]
            if not active_keys:
                continue

            window = [w - k for k in range(W, 0, -1)]  # [w-W, ..., w-1]
            for j in window:
                if j >= 0 and j not in zcache:
                    zcache[j] = model.encode_weeks(g, [j])[j]
            bank_w = {j: zcache[j] for j in window if j in zcache}

            samples: list[Sample] = []
            for (sid, chart) in active_keys:
                fsw = first_seen[(sid, chart)]
                window_weeks = [wk if (wk >= fsw and wk >= 0) else -1 for wk in window]
                pad_mask = [wk == -1 for wk in window_weeks]
                samples.append(
                    Sample(
                        song_idx=song_to_idx[sid],
                        chart=_CHART_CODE[chart],
                        target_week=w,
                        window_weeks=window_weeks,
                        pad_mask=pad_mask,
                        y=0.0,
                    )
                )

            with torch.no_grad():
                y_hat = model.predict(bank_w, samples)

            for samp, (sid, chart), pred in zip(samples, active_keys, y_hat.tolist()):
                work_bank[w, samp.song_idx, samp.chart] = pred
                key2 = (sid, chart, w)
                assert key2 in weekly_indexed.index, (
                    f"weekly_df is expected dense in [first_seen,last_seen] per (song,chart) "
                    f"(S3: ts_df is dense per day); missing {key2}"
                )
                y_true = float(weekly_indexed[key2])
                rows.append(
                    {"song_id": sid, "chart": chart, "week": w, "y_true": y_true, "y_pred": float(pred)}
                )
    finally:
        model.pop_bank = orig_bank

    return pd.DataFrame(rows, columns=["song_id", "chart", "week", "y_true", "y_pred"])


def gnn_rollout_recursive(
    model,
    g,
    weekly_df: pd.DataFrame,
    origins: pd.DataFrame,
    *,
    W: int,
    ks: tuple[int, ...] = (1, 2, 4),
    device: str = "cpu",
) -> pd.DataFrame:
    """Recursive GNN rollout (Mode 2): genuine k-step-ahead forecasting.

    For each origin ``(song_id, chart, week=w)``, the look-back window is
    anchored at the origin (``[w-W+1, ..., w]``) and its embeddings are
    encoded **once**, reused for every horizon — no future week is ever
    passed to ``encode_weeks`` (no structural information past the origin is
    available, by construction). The only thing that changes per horizon is
    the persistence anchor ``y_prev``: for ``k=1`` it is the real
    ``pop_bank[w]``; for ``k>=2`` it is the model's own prediction for
    ``w+k-1``, written into a working copy of ``pop_bank`` private to this
    origin (cloned fresh per origin so overlapping-song origins at different
    weeks never contaminate each other's "real" history).

    Args:
        model: a ``MusicDiffusionGNN`` with a non-None ``pop_bank`` buffer.
        g: the full ``hetero_full`` graph.
        weekly_df: output of ``aggregate_weekly`` (for ``y_true`` lookups and
            each (song,chart)'s ``first_seen`` week, for window padding).
        origins: DataFrame with columns ``[song_id, chart, week]``, one row
            per evaluation origin. Restricted internally to the test span
            (``week >= TEST_START_WEEK``) and to origins with room for the
            full horizon (``week + max(ks) <= 260``) — both filters applied
            defensively even if the caller already applied them.
        W: look-back window length (must match the model's trained config).
        ks: forecast horizons in weeks.
        device: torch device for model + graph.

    Returns:
        DataFrame with columns ``[song_id, chart, origin_week, k, y_true,
        y_pred]``, one row per ``(origin, k)``.
    """
    from music_diffusion_gnn.training.dataset import TEST_START_WEEK

    model = model.to(device)
    g = g.to(device)
    model.eval()
    assert model.pop_bank is not None, "gnn_rollout_recursive requires a model with a pop_bank buffer"

    song_ids = g["music"].song_id
    song_to_idx = {sid: i for i, sid in enumerate(song_ids)}
    first_seen = weekly_df.groupby(["song_id", "chart"])["week"].min().to_dict()
    weekly_indexed = weekly_df.set_index(["song_id", "chart", "week"])["y_week"]

    max_k = max(ks)
    n_weeks = model.pop_bank.shape[0]
    valid = (origins["week"] >= TEST_START_WEEK) & (origins["week"] + max_k <= n_weeks - 1)
    origins = origins.loc[valid, ["song_id", "chart", "week"]]

    orig_bank = model.pop_bank
    zcache: dict[int, torch.Tensor] = {}
    rows: list[dict] = []

    def _ensure_encoded(weeks: list[int]) -> None:
        missing = [j for j in weeks if j >= 0 and j not in zcache]
        if missing:
            model.pop_bank = orig_bank  # encode_weeks must only ever see real, historical data
            with torch.no_grad():
                zcache.update(model.encode_weeks(g, missing))

    try:
        for _, orow in origins.iterrows():
            sid, chart, w = orow["song_id"], orow["chart"], int(orow["week"])
            song_idx = song_to_idx[sid]
            chart_code = _CHART_CODE[chart]
            fsw = first_seen[(sid, chart)]

            window = [w - k for k in range(W - 1, -1, -1)]  # [w-W+1, ..., w], all <= w
            _ensure_encoded(window)
            bank_w = {j: zcache[j] for j in window if j in zcache}
            window_weeks = [wk if (wk >= fsw and wk >= 0) else -1 for wk in window]
            pad_mask = [wk == -1 for wk in window_weeks]

            work_bank = orig_bank.clone()  # isolated per origin: no cross-origin contamination
            model.pop_bank = work_bank
            for k in range(1, max_k + 1):
                target_week = w + k
                sample = Sample(
                    song_idx=song_idx,
                    chart=chart_code,
                    target_week=target_week,
                    window_weeks=window_weeks,
                    pad_mask=pad_mask,
                    y=0.0,
                )
                with torch.no_grad():
                    pred = float(model.predict(bank_w, [sample]).item())
                work_bank[target_week, song_idx, chart_code] = pred  # never a real value
                if k in ks:
                    key2 = (sid, chart, target_week)
                    y_true = float(weekly_indexed[key2]) if key2 in weekly_indexed.index else float("nan")
                    rows.append(
                        {
                            "song_id": sid,
                            "chart": chart,
                            "origin_week": w,
                            "k": k,
                            "y_true": y_true,
                            "y_pred": pred,
                        }
                    )
            model.pop_bank = orig_bank
    finally:
        model.pop_bank = orig_bank

    return pd.DataFrame(rows, columns=["song_id", "chart", "origin_week", "k", "y_true", "y_pred"])


def _saturation_rate(preds: Iterable[float], *, eps: float = 1e-9) -> float:
    """Fraction of predictions clamped at the floor (0) or ceiling (0.5).

    Diagnostic only (edge case in spec: report, don't mask). A high rate
    signals the rollout has diverged/saturated rather than tracking a
    plausible trajectory.
    """
    import numpy as np

    arr = np.asarray(list(preds), dtype=float)
    if arr.size == 0:
        return 0.0
    saturated = (arr <= eps) | (arr >= 0.5 - eps)
    return float(saturated.mean())
