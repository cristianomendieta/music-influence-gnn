"""Verdict of the attention phase, as pre-committed in ADR-0006 (Phase 7, ATT-05).

Pure functions over per-origin predictions (pandas/numpy, no torch); the heavy
run happens in the Colab notebook. Every value fixed by
``docs/adr/0006-precompromisso-fase-atencao.md`` lives here as a constant so the
notebooks import it instead of retyping it.
"""
from __future__ import annotations

import pandas as pd

from music_diffusion_gnn.evaluation.mapa import ALPHA
from music_diffusion_gnn.evaluation.stats import holm_correction, paired_by_song

# The variant under test (``A``) and its three references, one per hypothesis.
TREATED = "atencao"
CONTRASTS = {
    "H1": "sem_grafo",     # A beats the no-graph model (decides)
    "H2": "atencao_emb",   # A beats attention on the rewired graph (decides)
    "H3": "proposta",      # A beats the SAGE proposal (informative, does not decide)
}
CHARTS = ("viral50", "top200")
KS = (1, 2, 4)
N_CELLS = len(CHARTS) * len(KS)              # 6 cells per split
FAMILY_SIZE = N_CELLS * len(CONTRASTS)       # 18 comparisons per split, one Holm family
MAJORITY = 4                                 # "Maioria" = 4 of 6 cells

# What the dissertation says under each outcome (ADR-0006, table "O que a dissertação passa a dizer").
OUTCOMES = {
    "O1": "H1 e H2 em maioria: o nulo era do encoder; com atenção a topologia paga",
    "O2": "H1 em maioria, H2 não: o ganho vem de agregar atributos dos vizinhos, não da topologia",
    "O3": "H1 não está em maioria e A não perde em maioria: o nulo é robusto à arquitetura",
    "O4": "A perde para o sem_grafo em maioria: a atenção prejudica",
}
DISCORDANCE_NOTE = ("os splits discordam: a discordância é o resultado e a afirmação principal "
                    "fica restrita ao que se replica")

# Tuning budget (ADR-0006, "Ajuste de hiperparâmetros"): a declared limitation.
N_CONFIGS_ATTENTION = 8
N_CONFIGS_SAGE = 24
# Parameters of the SAGE proposal (W12_h128_l3). Same value as
# training.trainer.PARAM_BUDGET_REF, repeated so this module needs no torch.
SAGE_N_PARAMS = 453_121
# Tuning tie: val_mse differences up to this count as a tie and the config with
# fewer parameters wins (spec edge case; one between-seed std of item 06).
TIE_VAL_MSE = 0.000003

_KEYS = ["song_id", "chart", "origin_week", "k", "split_regime"]


def verdict(ci_lo: float, ci_hi: float, p_holm: float) -> str:
    """``vence``, ``perde`` or ``inconclusivo`` for one comparison of ``A`` against a reference.

    Same win/loss rule as :func:`music_diffusion_gnn.evaluation.mapa.label_cell`:
    ``vence`` is the whole bootstrap CI of the squared-error difference below zero
    **and** Holm ``p`` below 0.05; ``perde`` is the mirror. ADR-0006 has no
    equivalence margin, so everything else is ``inconclusivo``.
    """
    if ci_hi < 0 and p_holm < ALPHA:
        return "vence"
    if ci_lo > 0 and p_holm < ALPHA:
        return "perde"
    return "inconclusivo"


def compare_models(preds: pd.DataFrame, *, B: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """H1, H2 and H3 per split and per on-chart cell (chart × k), Holm over the 18 of each split.

    ``preds`` is long, one row per ``(modelo, origin)`` with ``y_true``, ``y_pred`` and
    ``onchart`` (the format of ``predicoes_cinco_modelos.parquet`` plus the attention
    models, seeds already averaged). Only on-chart rows enter. Each comparison is
    :func:`paired_by_song` on the squared errors of the origins both models share,
    so origins of the same song are averaged first and ``n_songs`` counts songs.
    ``mean_diff < 0`` means ``A`` is better.

    Raises ``ValueError`` if a model is missing from a split, or a split does not
    yield the 18 comparisons the ADR fixes.
    """
    models = [TREATED, *CONTRASTS.values()]
    d = preds[preds["onchart"] & preds["modelo"].isin(models)].copy()
    for split in sorted(preds["split_regime"].unique()):
        present = set(d.loc[d["split_regime"] == split, "modelo"])
        missing = [m for m in models if m not in present]
        if missing:
            raise ValueError(f"model(s) missing from the on-chart predictions of {split!r}: {missing}")
    d["err"] = (d["y_true"] - d["y_pred"]) ** 2
    wide = d.pivot_table(index=_KEYS, columns="modelo", values="err").reset_index()

    rows = []
    for (split, chart, k), grp in wide.groupby(["split_regime", "chart", "k"], observed=True):
        for hyp, control in CONTRASTS.items():
            pair = grp[["song_id", TREATED, control]].dropna()
            r = paired_by_song(pair[TREATED].to_numpy(), pair[control].to_numpy(),
                               pair["song_id"].to_numpy(), B=B, seed=seed)
            rows.append({"split_regime": split, "chart": chart, "k": k, "hipotese": hyp,
                         "contraste": control, "n_origens": len(pair), **r})
    table = pd.DataFrame(rows)

    for split, grp in table.groupby("split_regime"):
        if len(grp) != FAMILY_SIZE:
            raise ValueError(f"{split!r} has {len(grp)} comparisons, ADR-0006 fixes {FAMILY_SIZE}")
        table.loc[grp.index, "p_holm"] = holm_correction(grp["p_value"].to_numpy())
    table["rotulo"] = [verdict(r.ci_lo, r.ci_hi, r.p_holm) for r in table.itertuples()]
    return table


def classify_outcome(table: pd.DataFrame) -> pd.DataFrame:
    """Outcome O1 to O4 per split from a :func:`compare_models` table.

    O4 first (``A`` loses to ``sem_grafo`` in at least 4 of 6 cells); else O1 if H1
    and H2 win in at least 4; else O2 if only H1 does; else O3 (which includes H2
    without H1). H3 is counted but never decides. ``discordancia`` is True on every
    row when the splits disagree; ``afirmacao_principal`` is then
    ``"restrita ao que se replica"`` instead of the shared outcome code.
    """
    rows = []
    for split, grp in table.groupby("split_regime", sort=True):
        n = grp.groupby(["hipotese", "rotulo"]).size()
        h1, h2, h3 = (int(n.get((h, "vence"), 0)) for h in CONTRASTS)
        h1_loss = int(n.get(("H1", "perde"), 0))
        if h1_loss >= MAJORITY:
            code = "O4"
        elif h1 >= MAJORITY and h2 >= MAJORITY:
            code = "O1"
        elif h1 >= MAJORITY:
            code = "O2"
        else:
            code = "O3"
        rows.append({"split_regime": split, "vitorias_h1": h1, "derrotas_h1": h1_loss,
                     "vitorias_h2": h2, "vitorias_h3": h3,
                     "desfecho": code, "afirmacao": OUTCOMES[code]})
    out = pd.DataFrame(rows)
    discord = out["desfecho"].nunique() > 1
    out["discordancia"] = discord
    out["afirmacao_principal"] = "restrita ao que se replica" if discord else out["desfecho"]
    out["nota"] = DISCORDANCE_NOTE if discord else ""
    return out


_CELL = ["split_regime", "chart", "k", "hipotese", "contraste"]


def sensitivity_by_seed(
    preds_seeds: pd.DataFrame, main_table: pd.DataFrame, **compare_kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat H1 to H3 on every isolated seed and compare with the test on the seed mean.

    ``preds_seeds`` is the format of :func:`compare_models` plus a ``seed`` column
    (one run per seed, not the average). ``main_table`` is the
    :func:`compare_models` table on the seed mean. Returns ``(cells, outcomes)``:
    ``cells`` has the main label of each cell, the label of every seed
    (``rotulo_seed{s}``), how many seeds agree (``seeds_concordam``) and how many
    had the cell (``n_seeds``); ``outcomes`` has, per split, the outcome of the
    mean (``desfecho_media``) next to each seed's (``desfecho_seed{s}``) and how
    many seeds agree with it.
    """
    cells = main_table[[*_CELL, "rotulo"]].copy()
    outcomes = classify_outcome(main_table)[["split_regime", "desfecho"]].rename(
        columns={"desfecho": "desfecho_media"})
    for s, grp in preds_seeds.groupby("seed", sort=True):
        t = compare_models(grp.drop(columns="seed"), **compare_kwargs)
        cells = cells.merge(t[[*_CELL, "rotulo"]].rename(columns={"rotulo": f"rotulo_seed{s}"}),
                            on=_CELL, how="left")
        o = classify_outcome(t)[["split_regime", "desfecho"]]
        outcomes = outcomes.merge(o.rename(columns={"desfecho": f"desfecho_seed{s}"}),
                                  on="split_regime", how="left")
    label_cols = [c for c in cells.columns if c.startswith("rotulo_seed")]
    cells["seeds_concordam"] = cells[label_cols].eq(cells["rotulo"], axis=0).sum(axis=1)
    cells["n_seeds"] = cells[label_cols].notna().sum(axis=1)
    out_cols = [c for c in outcomes.columns if c.startswith("desfecho_seed")]
    outcomes["seeds_concordam"] = outcomes[out_cols].eq(outcomes["desfecho_media"], axis=0).sum(axis=1)
    return cells, outcomes


def _pt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def budget_block(n_params_attention: int, *, n_params_sage: int = SAGE_N_PARAMS) -> str:
    """The parameter and tuning budget text that goes next to the results table (AC6)."""
    diff = (n_params_attention - n_params_sage) / n_params_sage
    return (
        f"Parâmetros: A = {_pt_int(n_params_attention)}; proposta SAGE = "
        f"{_pt_int(n_params_sage)} ({diff:+.1%}).\n"
        f"Orçamento de ajuste: {N_CONFIGS_ATTENTION} contra {N_CONFIGS_SAGE} configurações "
        f"(a proposta SAGE teve {N_CONFIGS_SAGE}, a atenção {N_CONFIGS_ATTENTION}): "
        "limitação declarada, não vantagem escondida."
    )
