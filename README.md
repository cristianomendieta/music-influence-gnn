# Music Diffusion GNN — Replicação + Extensão (BraSNAM 2026)

Replicação de Oliveira et al. 2025 (BraSNAM) e extensão para Temporal GNN heterogêneo
sobre o grafo artista–música–gênero da cena musical brasileira no Spotify.

Documento principal de pesquisa: [PLANO.md](PLANO.md).
Planejamento operacional (specs, tasks, estado): [.specs/](.specs/).

## Estrutura

```
data/                 # Datasets (MGD+ charts + features)
  charts/mgdplus/     #   Top 200 + Viral 50 BR diários, 2017–2022 (completo)
  songs/              #   Features acústicas + metadata por ano
  artists/            #   Artistas + listas de gêneros
  genre_network/      #   Rede gênero↔gênero pré-construída
  MGDplus/            #   Dataset MGD+ completo (68 mercados) — preservado

src/music_diffusion_gnn/   # Código importável (pip install -e .)
  data/               #   loaders, pré-processamento (rank score, MA-7d)
  baselines/          #   sir.py (baseline SIR BraSNAM 2025)
  graph/              #   construção do HeteroData (PyG)
  models/             #   HeteroSAGE + GRU
  training/           #   Lightning module, splits temporais
  evaluation/         #   métricas, Mann-Whitney, plots

scripts/              # Entrypoints CLI
  run_phase0.py       #   pipeline Phase 0 (subset → timeseries → SIR → report)

notebooks/            # Exploração e geração de figuras
results/              # Artifacts (gitignored)
references/           # PDFs dos papers citados
```

## Setup

```bash
pip install -e .[dev]
python scripts/run_phase0.py --force
```

Saída esperada: 1.981 músicas na interseção viral∩hit, SIR converge em ~60s, summary.md gerado em results/phase0/.

## Citações obrigatórias no paper final

- **Features e gêneros**: Seufitelli, D. B.; Oliveira, G. P.; Silva, M. O.; Moro, M. M. *MGD+: An Enhanced Music Genre Dataset with Success-based Networks.* DSW 2023.
- **Paper original (replicação)**: Oliveira, G. P.; Vassio, L.; Couto da Silva, A. P.; Moro, M. M. *Modeling music popularity as an epidemic.* BraSNAM 2025.
