"""EDA — sustains or rejects the median-imputation decision for missing acoustic
attributes (docs.md issue #11).

Question: is missingness in the 9 acoustic columns (ACOUSTIC_COLS) roughly
random, and do songs missing acoustic features have a different popularity
profile than songs with complete features? If missingness correlates with
popularity, imputing with a single central value (mean/median) for all of
them biases the node features toward "typical" acoustic style regardless of
how popular the song actually was.

Uses the graph's actual node universe (`data/processed/graph/node_id_map.json`,
the ground truth for what `build_music_nodes` produced) rather than
re-deriving it from the raw chart/song CSVs on disk — those CSVs turned out
to cover only 5,010 of the graph's 6,526 music nodes (the graph was built
from a fuller MGD+ snapshot than what ships in `data/songs/`), so rebuilding
the universe from them undercounts missingness by two orders of magnitude.

Usage: python scripts/exploratory/11_eda_imputacao_mediana.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from music_diffusion_gnn.data.loaders import load_charts, load_songs
from music_diffusion_gnn.graph.nodes import ACOUSTIC_COLS

OUT = ROOT / "results" / "eda_imputacao_mediana.md"
NODE_ID_MAP = ROOT / "data" / "processed" / "graph" / "node_id_map.json"


def main() -> None:
    songs_df = load_songs()
    # `timeseries.parquet` only covers the viral∩hit evaluation subset (~1,980
    # songs), not the graph's full 6,526-node universe — using it here would
    # silently treat "outside that subset" as "zero popularity" for the very
    # songs missing acoustic features (they're disjoint from the eval subset).
    # The raw MGD+ chart appearances cover the whole universe instead.
    charts_df = load_charts()

    with open(NODE_ID_MAP) as f:
        nmap = json.load(f)
    universe = sorted(nmap["music"]["spotify_id_to_idx"].keys())
    N = len(universe)

    songs_indexed = songs_df.set_index("song_id")
    acoustic = songs_indexed.reindex(universe)[ACOUSTIC_COLS]
    missing_any = acoustic.isna().any(axis=1)
    n_missing = int(missing_any.sum())

    # --- 1. Missingness by attribute -----------------------------------
    per_attr = (acoustic.isna().mean() * 100).round(2)

    # --- 2. Missingness by year ------------------------------------------
    # release_date only exists for the 5,010 songs that DO have a songs_df
    # row — useless for a by-year breakdown of missingness, since it can
    # never cover the very rows that are missing. Year of first chart
    # appearance (from `charts_df`, which spans the full universe) is used
    # instead.
    first_chart_year = (
        charts_df.groupby("song_id")["date"].min().dt.year.reindex(universe)
    )
    by_year = (
        pd.DataFrame({"missing": missing_any, "year": first_chart_year})
        .groupby("year")["missing"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "pct_missing", "size": "n_songs"})
    )
    by_year["pct_missing"] = (by_year["pct_missing"] * 100).round(1)
    no_songs_row = songs_indexed.reindex(universe).index.difference(songs_indexed.index)

    # --- 3. Popularity profile: missing vs complete ----------------------
    # Chart-appearance days per song, across both charts (raw rows = days ranked;
    # unlike timeseries.parquet, `charts_df` is not calendar-dense, so a row only
    # exists on days the song was actually on a chart — no floor/zero-fill to undo).
    doc_by_song = charts_df.groupby("song_id").size()

    pop_missing = doc_by_song.reindex([s for s in universe if missing_any.loc[s]]).fillna(0)
    pop_complete = doc_by_song.reindex([s for s in universe if not missing_any.loc[s]]).fillna(0)

    u_stat, p_value = stats.mannwhitneyu(pop_missing, pop_complete, alternative="two-sided")

    # ponytail: cheap sanity checks, not a full test suite — catches a broken
    # join or an empty group silently producing a meaningless p-value.
    assert len(pop_missing) + len(pop_complete) == N
    assert 0.0 <= per_attr.max() <= 100.0
    assert not np.isnan(p_value)

    # --- Write report -----------------------------------------------------
    lines = [
        "# EDA — imputação por mediana dos atributos acústicos ausentes",
        "",
        f"Universo (idêntico a `build_music_nodes`): {N} músicas "
        f"({n_missing} com ao menos um atributo acústico ausente, "
        f"{100 * n_missing / N:.1f}%).",
        "",
        "## 1. Ausência por atributo",
        "",
        "| Atributo | % ausente |",
        "|----------|-----------|",
    ]
    for col, pct in per_attr.items():
        lines.append(f"| {col} | {pct:.2f}% |")

    lines += ["", "## 2. Ausência por ano de primeira aparição em chart", "",
              "| Ano | % ausente | N músicas |",
              "|-----|-----------|-----------|"]
    for year, row in by_year.sort_index().iterrows():
        y = "desconhecido" if pd.isna(year) else int(year)
        lines.append(f"| {y} | {row['pct_missing']:.1f}% | {int(row['n_songs'])} |")
    if len(no_songs_row) > 0:
        lines.append(
            f"\n{len(no_songs_row)} músicas do universo não têm nenhuma linha em `songs_df` "
            "(ausência total, contam como 'ausente' em todos os atributos e como ano desconhecido)."
        )

    lines += [
        "",
        "## 3. Perfil de popularidade: ausente vs. completo",
        "",
        "Dias de aparição em chart (linhas do MGD+ com posição registrada, somadas entre Top 200 e Viral 50):",
        "",
        f"- Ausente ({len(pop_missing)} músicas): média={pop_missing.mean():.1f}, "
        f"mediana={pop_missing.median():.1f}",
        f"- Completo ({len(pop_complete)} músicas): média={pop_complete.mean():.1f}, "
        f"mediana={pop_complete.median():.1f}",
        f"- Mann-Whitney U: p={p_value:.3e}",
        "",
        "## Conclusão",
        "",
    ]

    if p_value < 0.05:
        lines += [
            "A ausência de atributos acústicos **não é aproximadamente aleatória** em relação à "
            "popularidade (diferença de mediana de dias em chart estatisticamente significativa, "
            f"p={p_value:.1e}). A imputação atual por um único valor central (mean/median, "
            "indistinto do restante da distribuição) **não se sustenta sem ressalva**: ela empurra "
            "as músicas ausentes para o centro do espaço de atributos acústicos independentemente "
            "de seu perfil real de popularidade, o que pode diluir sinal correlacionado a esse "
            "perfil. O indicador binário de ausência (já presente no vetor de nó) mitiga parcialmente "
            "o problema, permitindo ao modelo aprender a tratar esses casos separadamente — mas não "
            "substitui um valor imputado mais informativo.",
            "",
            "**Alternativa proposta:** imputar por classe de popularidade (ex.: mediana condicionada "
            "a faixas de dias-em-chart, ou ao chart de origem) em vez de um único valor central global, "
            "preservando o indicador de ausência. Como o dataset já carrega esse indicador, o risco "
            "prático do viés é parcialmente absorvido pelo modelo — mas a imputação condicional é a "
            "correção recomendada para a versão final.",
        ]
    else:
        lines += [
            "A ausência de atributos acústicos é **aproximadamente independente** do perfil de "
            "popularidade das músicas (sem diferença estatisticamente significativa de dias em "
            f"chart entre os grupos, p={p_value:.2f}). A imputação por um valor central único, "
            "acompanhada do indicador binário de ausência, **se sustenta**: não há evidência de que "
            "ela introduza viés sistemático relacionado à popularidade.",
            "",
            "Nota: apesar de o texto da metodologia e o comentário do código chamarem a imputação de "
            "\"mediana\", a implementação em `graph/nodes.py` (`_zscore`/`acoustic_z`) atribui às "
            "linhas ausentes o valor `0.0` do z-score, isto é, a **média** (não a mediana) da "
            "distribuição observada. Isso não muda a conclusão desta EDA sobre o mecanismo de "
            "ausência, mas é uma imprecisão de nomenclatura a corrigir no texto e no comentário.",
        ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
