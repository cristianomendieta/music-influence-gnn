# Phase 1 — Construção do grafo heterogêneo

**Status:** implemented
**Janela:** semanas 2–3 (2026-05-17 → 2026-05-31)
**Depende de:** Phase 0 (`subset_ids.json`, `timeseries.parquet`)
**Bloqueia:** Phases 2–4

## Goal

HeteroData PyG com 3 tipos de nó (music, artist, genre) e 4 tipos de aresta,
+ estatísticas exploratórias, prontos para o Temporal GNN do Phase 2 sem
retrabalho. Sem leakage temporal: cada aresta carrega `first_seen_week` para
mascaramento em runtime.

## Out of scope

- Treino de qualquer GNN (Phase 2).
- Splits de train/val/test (Phase 2).
- Features de séries temporais (já no parquet do Phase 0).
- Predição genuína k > 0 (Phase 3).

## Requirements

### R0 — Nós (3 tipos)

- **R0.1** Music nodes: **6.469** IDs (universo Top200 BR 2017-2021). Features iniciais:
  9 acústicas (MGD+) + popularity + explicit + song_type + total_streams + dias_no_chart.
  Float32, normalizadas (z-score por feature contínua).
- **R0.2** Artist nodes: **1.701** IDs. Features: num_hits, num_collab_hits, anos_no_chart, n_genres.
- **R0.3** Genre nodes: **530** IDs. Features: one-hot ou embedding aprendido — escolha em design.md.
  Spec só exige que `x` seja tensor `(N_g, d)` com `d ≥ 1`.
- **R0.4** IDs consistentes entre nó e séries: todo `song_id` do subset (1.981) deve existir
  em music nodes; chave de junção é o `song_id` Spotify.
- **R0.5** Sem NaN nas features (imputar mediana se necessário; flag de imputação opcional).

### R1 — Arestas (4 tipos)

- **R1.1** `(artist, performs, music)` — direcionada. Atributos: `role ∈ {main, feat}`,
  `position_in_list` (int). Fonte: lista `artist_id` em `br-hit_songs-*.csv`.
- **R1.2** `(artist, has_genre, genre)` — não-direcionada (PyG: duas direções).
  Fonte: lista `genres` em `br-artists-all_time.csv`. Sem features.
- **R1.3** `(music, cotrajectory, music)` — direcionada (ordem de entrada no chart).
  Atributos: `days_together` (int ≥ 7), `avg_position_distance` (float),
  `chart ∈ {viral50, top200}` (int categórico), `first_seen_week` (int).
  Construção: para cada par (i, j) que coexistiu ≥ 7 dias no chart C,
  cria aresta i→j se i entrou antes de j (data de primeira aparição no chart).
- **R1.4** `(genre, cooccurs, genre)` — não-direcionada. Atributos: `weight`, `avg_popularity`,
  `avg_streams`, `first_seen_week` (derivado do ano do arquivo MGD+ que introduziu a aresta).
  Fonte: `data/MGDplus/genre_network/br/*.csv` (arquivos anuais 2017–2021).
- **R1.5** Toda aresta DEVE ter `first_seen_week ∈ [0, 260]` (semana zero = 2017-W1) para
  permitir mascaramento temporal sem leakage no Phase 2.

### R2 — Temporal (snapshots semanais via máscara)

- **R2.1** Calendário canônico: ISO week (ano-semana), 2017-W1 → 2021-W52 (~260 semanas).
  Helper `week_index(date) → int` em `src/music_diffusion_gnn/graph/temporal.py`.
- **R2.2** Função `mask_until(hetero, week_t) → HeteroData` retorna um clone com
  apenas as arestas cujo `first_seen_week ≤ week_t`. Sem cópia profunda de features.
- **R2.3** Snapshots NÃO são materializados em disco. Apenas o grafo full + máscara.

### R3 — Saídas persistidas

- **R3.1** `data/processed/graph/hetero_full.pt` — HeteroData serializado via `torch.save`.
- **R3.2** `data/processed/graph/node_id_map.json` — mapeia índice PyG ↔ ID original
  (`spotify_song_id`, `artist_id`, `genre_name`) para cada tipo. Reprodutibilidade.
- **R3.3** `results/phase1/stats.md` — relatório markdown com:
  distribuição de grau por tipo de nó/aresta (média, mediana, p95, max);
  número de componentes conexas (por subgrafo homogêneo); clustering coefficient;
  top-10 comunidades por gênero (Louvain via networkx);
  comparação de contagens com o PLANO (6.469 / 1.701 / 530 ± δ).
- **R3.4** `results/phase1/degree_distributions.png` — figura com 4 painéis (um por tipo de aresta).

### R4 — Critérios de aceitação

A fase só é concluída se TODOS abaixo passarem:

| # | Critério | Tolerância |
|---|---|---|
| C1 | n_music = 6.469 | ±100 (universo real = top200 ∪ viral50-with-features = 6.526) |
| C2 | n_artist = 1.701 | ±5 |
| C3 | n_genre = 530 | ±10 |
| C4 | Toda música do subset (1.981) está no grafo | 0 violação |
| C5 | Todos os artistas das músicas do subset estão alcançáveis | 0 isolados |
| C6 | Nenhuma aresta aponta para nó inexistente | 0 violação |
| C7 | Nenhuma aresta com `first_seen_week ∉ [0, 260]` | 0 violação |
| C8 | Smoke-test: HeteroSAGE (2 camadas, hidden=128) forward em CPU sem erro | passa |
| C9 | `mask_until(grafo, week=130)` reduz arestas monotonicamente vs `week=260` | passa |

### R5 — Reprodutibilidade

- **R5.1** Pipeline regenerável com `python scripts/run_phase1.py`.
- **R5.2** Sem aleatoriedade nesta fase (graph build é determinístico).
- **R5.3** Versões de torch / torch_geometric / networkx fixadas em `pyproject.toml`.

## Acceptance test

1. `python scripts/run_phase1.py` roda end-to-end sem erro (< 10 min em laptop).
2. Critérios C1–C9 todos verdes; smoke-test imprime shape do embedding `(6469, 128)`.
3. `results/phase1/stats.md` existe e contém as 4 distribuições + comunidades Louvain.
4. PR único e atômico referenciando spec/design/tasks.

## Open questions (a resolver no design.md)

- Features de gênero: one-hot (530-d) ou embedding aprendido (e.g., 32-d)?
- Quando duas músicas coexistem em ambos os charts ≥ 7 dias, criar 1 aresta com
  `chart=union` ou 2 arestas paralelas (uma por chart)?
- Política para músicas sem features acústicas no MGD+ (subset de 6.469 não cobre tudo):
  imputar, dropar do grafo, ou flag `acoustic_missing`?
- Direção da aresta `has_genre`: PyG exige ambas direções explícitas; modelar como
  `(artist, has_genre, genre)` + reverso, ou usar `ToUndirected()`?
- Embedding inicial de gênero precisa de pretrain (e.g., node2vec) ou só identidade?

## Traceability

- Phase 1 spec ↔ PLANO.md §4 (Fase 1) + ROADMAP.md (linhas 33–47).
- R0/R1 contagens ↔ Tabela "Schema" em PLANO.md linhas 87–91.
- R2 (semanais) ↔ PLANO.md linha 105 ("Snapshots semanais para o componente temporal").
- R3.3 conteúdo ↔ ROADMAP linha 46 ("estatísticas exploratórias + HeteroData serializado").
