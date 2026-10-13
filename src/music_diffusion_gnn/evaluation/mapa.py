"""Per-stratum verdicts of the structure map (Phase 6b).

Pure functions over per-origin squared errors; the heavy run (10 000 bootstrap
resamples over dozens of cells) happens in the Colab notebook. Definitions in
``.specs/features/phase-6b-mapa-estrutura/spec.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from music_diffusion_gnn.evaluation.stats import (
    bootstrap_ci_diff,
    holm_correction,
    wilcoxon_signed_rank,
)


def compare_cell(
    err_a: np.ndarray,
    err_b: np.ndarray,
    song_ids: np.ndarray,
    *,
    B: int = 10_000,
    seed: int,
    equiv_frac: float = 0.05,
) -> dict[str, float]:
    """Paired comparison of two variants on the origins of one cell.

    ``err_a`` / ``err_b`` are squared errors per origin (a = the proposal,
    b = the control). Origins are averaged per song first, so a song that appears
    in both charts, or with many origins, is one observation. ``mean_diff < 0``
    means ``a`` is better. ``margin`` is the equivalence half-width:
    ``equiv_frac`` of the control's mean per-song squared error.
    """
    df = pd.DataFrame({
        "song_id": np.asarray(song_ids),
        "a": np.asarray(err_a, dtype=float),
        "b": np.asarray(err_b, dtype=float),
    })
    per_song = df.groupby("song_id", observed=True)[["a", "b"]].mean()
    a = per_song["a"].to_numpy()
    b = per_song["b"].to_numpy()

    if np.array_equal(a, b):
        # scipy's Wilcoxon raises when every difference is zero; no difference is p = 1
        w = {"statistic": 0.0, "p_value": 1.0, "n": len(a)}
    else:
        w = wilcoxon_signed_rank(a, b)
    lo, hi, md = bootstrap_ci_diff(a, b, B=B, seed=seed)
    mean_b = float(b.mean())
    return {
        "mean_diff": md,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": w["p_value"],
        "win_rate": float((a < b).mean()),
        "n_songs": len(per_song),
        "mean_b": mean_b,
        "margin": equiv_frac * mean_b,
    }


ALPHA = 0.05


def label_cell(ci_lo: float, ci_hi: float, p_holm: float, margin: float) -> str:
    """Verdict of one tested cell, in this order of precedence.

    ``vence``: the whole CI of the squared-error difference is below zero and the
    Holm ``p`` is below 0.05 (``perde`` is the mirror). Otherwise ``equivalente``
    if the CI fits inside ``±margin``, else ``inconclusivo``. A difference that is
    both significant and inside the margin is reported as a win or a loss: the
    margin is only for the cells that show no effect.
    """
    if ci_hi < 0 and p_holm < ALPHA:
        return "vence"
    if ci_lo > 0 and p_holm < ALPHA:
        return "perde"
    if ci_lo >= -margin and ci_hi <= margin:
        return "equivalente"
    return "inconclusivo"


def holm_family(tests: pd.DataFrame) -> pd.Series:
    """Holm-adjusted ``p`` with one family per split, both contrasts together.

    ``tests`` needs ``split_regime``, ``p_value`` and ``insuficiente``. Cells
    marked ``insuficiente`` are outside the family (their ``p_holm`` is NaN) and
    do not count in the correction of the others.
    """
    out = pd.Series(np.nan, index=tests.index, name="p_holm")
    for _, grp in tests[~tests["insuficiente"]].groupby("split_regime"):
        out.loc[grp.index] = holm_correction(grp["p_value"].to_numpy())
    return out
