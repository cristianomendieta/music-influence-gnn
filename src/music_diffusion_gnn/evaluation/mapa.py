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


_KEYS = ["song_id", "chart", "origin_week", "k", "split_regime"]


def build_map(
    preds: pd.DataFrame,
    strata: pd.DataFrame,
    *,
    treated: str = "proposta",
    controls: tuple[str, ...] = ("sem_grafo", "embaralhado"),
    B: int = 10_000,
    seed: int = 42,
    min_songs: int = 30,
    equiv_frac: float = 0.05,
) -> pd.DataFrame:
    """Per-stratum verdict table: one row per ``(split_regime, estrato, k, contraste)``.

    ``preds`` is long, one row per ``(model, origin)`` with ``y_true``, ``y_pred`` and
    ``onchart`` (the format of ``predicoes_cinco_modelos.parquet``); only the
    on-chart rows of ``treated`` and ``controls`` are used. ``strata`` maps
    ``(song_id, chart, origin_week)`` to ``estrato``; origins without a stratum or
    marked ``sem_serie`` are left out. Cells with fewer than ``min_songs`` songs
    get ``rotulo = "insuficiente"`` and no statistics.
    """
    models = [treated, *controls]
    d = preds[preds["onchart"] & preds["modelo"].isin(models)]
    d = d.merge(strata[["song_id", "chart", "origin_week", "estrato"]],
                on=["song_id", "chart", "origin_week"], how="inner")
    d = d[d["estrato"] != "sem_serie"].copy()
    d["err"] = (d["y_true"] - d["y_pred"]) ** 2
    wide = d.pivot_table(index=[*_KEYS, "estrato"], columns="modelo", values="err").dropna().reset_index()

    rows = []
    for (split, estrato, k), grp in wide.groupby(["split_regime", "estrato", "k"], observed=True):
        n_songs = grp["song_id"].nunique()
        for control in controls:
            base = {"split_regime": split, "estrato": estrato, "k": k, "contraste": control,
                    "n_origens": len(grp), "n_musicas": n_songs, "insuficiente": n_songs < min_songs}
            if base["insuficiente"]:
                rows.append(base)
                continue
            r = compare_cell(grp[treated].to_numpy(), grp[control].to_numpy(),
                             grp["song_id"].to_numpy(), B=B, seed=seed, equiv_frac=equiv_frac)
            rows.append({**base, **r})

    table = pd.DataFrame(rows)
    table["p_holm"] = holm_family(table)
    tested = ~table["insuficiente"]
    table["rotulo"] = "insuficiente"
    table.loc[tested, "rotulo"] = [
        label_cell(r.ci_lo, r.ci_hi, r.p_holm, r.margin) for r in table[tested].itertuples()
    ]
    return table


def structure_pays(table: pd.DataFrame, *, controls: tuple[str, ...] = ("sem_grafo", "embaralhado"),
                   min_horizons: int = 2) -> pd.DataFrame:
    """Which strata the structure pays in: ``estrutura paga``, ``nao paga`` or ``nao testavel``.

    A horizon counts for a split when the treated model ``vence`` **both** controls
    at that ``k``. A stratum is ``estrutura paga`` when that holds for at least
    ``min_horizons`` horizons in **every** split. It is ``nao testavel`` when some
    split has fewer than ``min_horizons`` testable horizons (all controls
    sufficiently sampled), since the criterion cannot be met there.
    """
    out = []
    splits = sorted(table["split_regime"].unique())
    for estrato, grp in table.groupby("estrato", observed=True):
        per_split = {}
        for split in splits:
            by_k = grp[grp["split_regime"] == split].groupby("k", observed=True)
            testable = sum(
                (len(g) == len(controls)) and bool((g["rotulo"] != "insuficiente").all())
                for _, g in by_k
            )
            wins = sum(bool((g["rotulo"] == "vence").sum() == len(controls)) for _, g in by_k)
            per_split[split] = (testable, wins)
        if any(t < min_horizons for t, _ in per_split.values()):
            verdict = "nao testavel"
        elif all(w >= min_horizons for _, w in per_split.values()):
            verdict = "estrutura paga"
        else:
            verdict = "nao paga"
        out.append({"estrato": estrato, "veredito": verdict,
                    **{f"horizontes_vencidos_{s}": w for s, (_, w) in per_split.items()}})
    return pd.DataFrame(out)


_CELL = ["split_regime", "estrato", "k", "contraste"]


def sensitivity_by_seed(
    preds_seeds: pd.DataFrame, strata: pd.DataFrame, main_map: pd.DataFrame, **build_kwargs
) -> pd.DataFrame:
    """Does each label of ``main_map`` repeat when the map is rebuilt seed by seed?

    ``preds_seeds`` is the format of :func:`build_map` plus a ``seed`` column (one
    model run per seed, not the seed average). Returns ``main_map``'s cells with
    the label of every seed, how many seeds agree with the main label
    (``seeds_concordam``) and how many seeds had the cell (``n_seeds``).
    """
    per_seed = []
    for seed, grp in preds_seeds.groupby("seed"):
        m = build_map(grp, strata, **build_kwargs)[[*_CELL, "rotulo"]]
        per_seed.append(m.rename(columns={"rotulo": f"rotulo_seed{seed}"}))
    out = main_map[[*_CELL, "rotulo"]].copy()
    for m in per_seed:
        out = out.merge(m, on=_CELL, how="left")
    seed_cols = [c for c in out.columns if c.startswith("rotulo_seed")]
    out["seeds_concordam"] = out[seed_cols].eq(out["rotulo"], axis=0).sum(axis=1)
    out["n_seeds"] = out[seed_cols].notna().sum(axis=1)
    return out


def by_chart_rmse(
    preds: pd.DataFrame, strata: pd.DataFrame, *, treated: str = "proposta",
    controls: tuple[str, ...] = ("sem_grafo", "embaralhado"),
) -> pd.DataFrame:
    """RMSE difference (treated − control) per stratum and chart. Descriptive: no test, no label."""
    d = preds[preds["onchart"] & preds["modelo"].isin([treated, *controls])]
    d = d.merge(strata[["song_id", "chart", "origin_week", "estrato"]],
                on=["song_id", "chart", "origin_week"], how="inner")
    d = d[d["estrato"] != "sem_serie"].copy()
    d["err"] = (d["y_true"] - d["y_pred"]) ** 2
    mse = (d.groupby(["split_regime", "estrato", "chart", "k", "modelo"], observed=True)["err"]
             .agg(["mean", "size"]).reset_index())
    base = mse[mse["modelo"] == treated].rename(columns={"mean": "mse_tratado", "size": "n_origens"})
    out = []
    for control in controls:
        c = mse[mse["modelo"] == control][["split_regime", "estrato", "chart", "k", "mean"]]
        m = base.merge(c.rename(columns={"mean": "mse_controle"}),
                       on=["split_regime", "estrato", "chart", "k"])
        m["contraste"] = control
        m["dif_rmse"] = np.sqrt(m["mse_tratado"]) - np.sqrt(m["mse_controle"])
        out.append(m[["split_regime", "estrato", "chart", "k", "contraste", "n_origens", "dif_rmse"]])
    return pd.concat(out, ignore_index=True)


def margin_sensitivity(table: pd.DataFrame, fracs: tuple[float, ...] = (0.025, 0.10)) -> pd.DataFrame:
    """Cells whose label changes when the equivalence margin is ``fracs`` of the control's MSE.

    The bootstrap does not depend on the margin, so labels are recomputed from the
    stored CI and ``mean_b``. Returns only the tested cells whose label differs
    from the main one for some fraction.
    """
    t = table[~table["insuficiente"]].copy()
    changed = pd.Series(False, index=t.index)
    for f in fracs:
        col = f"rotulo_{f:g}"
        t[col] = [label_cell(r.ci_lo, r.ci_hi, r.p_holm, f * r.mean_b) for r in t.itertuples()]
        changed |= t[col] != t["rotulo"]
    cols = [*_CELL, "rotulo", *[f"rotulo_{f:g}" for f in fracs]]
    return t.loc[changed, cols].reset_index(drop=True)
