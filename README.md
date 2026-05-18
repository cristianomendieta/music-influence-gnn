# Music Diffusion GNN — Replicação + Extensão (BraSNAM 2026)

Replicação de Oliveira et al. 2025 (BraSNAM) e extensão para Temporal GNN heterogêneo
sobre o grafo artista–música–gênero da cena musical brasileira no Spotify.

**Status atual:** Phase 0 (baseline SIR) concluída ✅ — ver [STATUS.md](STATUS.md).
**Planejamento:** [.specs/](.specs/) | **Visão de pesquisa:** [PLANO.md](PLANO.md).

## Estrutura

```
data/                 # Datasets (MGD+ charts + features) — gitignored
  charts/mgdplus/     #   Top 200 + Viral 50 BR diários, 2017–2022 (completo)
  songs/              #   Features acústicas + metadata por ano
  artists/            #   Artistas + listas de gêneros
  genre_network/      #   Rede gênero↔gênero pré-construída
  MGDplus/            #   Dataset MGD+ completo (68 mercados) — preservado

src/music_diffusion_gnn/   # Código importável (pip install -e .)
  data/               #   loaders, pré-processamento (rank score → MA-7d → min-max)
  baselines/          #   sir.py (baseline SIR, Phase 0 ✅)
  graph/              #   construção do HeteroData PyG (Phase 1, stub)
  models/             #   HeteroSAGE + GRU (Phase 2, stub)
  training/           #   Lightning module, splits temporais (Phase 2, stub)
  evaluation/         #   métricas, Mann-Whitney, plots

scripts/
  run_phase0.py             # Pipeline Phase 0 — entry point principal
  exploratory/              # Diagnósticos e validação de dados (não fazem parte do pipeline)
    verify_data.py
    inspect_mgdplus.py

exploration/          # Notebooks EDA pré-Phase 1 (00_overview → 06_gnn_design_sketch)
notebooks/            # Figuras e análises (fases futuras)
results/              # Artefatos gerados (gitignored; Phase 0 em results/phase0/)
references/           # PDFs dos papers citados
```

## Setup

```bash
pip install -e .[dev]
```

Phase 0 já foi executada. Para reproduzir (idempotente — usa cache se disponível):

```bash
python scripts/run_phase0.py
```

Saída em `results/phase0/`: `summary.md` (5/5 critérios ✅), `sir_params.parquet`, `boxplot_fig3.png`.

Para rodar os testes:

```bash
pytest tests/ -v
```

## Citações obrigatórias no paper final

- **Features e gêneros**: Seufitelli, D. B.; Oliveira, G. P.; Silva, M. O.; Moro, M. M. *MGD+: An Enhanced Music Genre Dataset with Success-based Networks.* DSW 2023.
- **Paper original (replicação)**: Oliveira, G. P.; Vassio, L.; Couto da Silva, A. P.; Moro, M. M. *Modeling music popularity as an epidemic.* BraSNAM 2025.
