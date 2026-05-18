# Phase 1 — Tasks (Construção do grafo heterogêneo)

**Spec:** [`spec.md`](spec.md)
**Design:** [`design.md`](design.md)
**Status:** Implemented (2026-05-17)

---

## Execution Plan

### Wave 1 — Foundation (sequencial)

Helpers temporais e dependências. Bloqueia tudo o resto.

```
T1 → T2
```

### Wave 2 — Builders (paralelos)

7 builders independentes. Todos consomem CSVs/parquet, retornam tensors. Não tocam em PyG.

```
T2 ──┬→ T3 [P] ─┐
     ├→ T4 [P] ─┤
     ├→ T5 [P] ─┤
     ├→ T6 [P] ─┼─→ T10
     ├→ T7 [P] ─┤
     ├→ T8 [P] ─┤
     └→ T9 [P] ─┘
```

### Wave 3 — Assembly & validação (sequencial)

Orquestrador monta `HeteroData`, aplica `ToUndirected`, valida C1-C7, serializa.

```
T3..T9 → T10 → T11
```

### Wave 4 — Stats & relatório (paralelos após T11)

Métricas e visualizações lêem o `.pt` serializado.

```
T11 ──┬→ T12 [P]
      └→ T13 [P]
```

### Wave 5 — Entrypoint + smoke test (sequencial)

CLI orquestra build + stats + smoke-test (C8/C9).

```
T11, T12, T13 → T14
```

### Wave 6 — Testes unitários (paralelos)

Mínimos: temporal puro + sanidade dos builders.

```
T1 → T15 [P]
T10 → T16 [P]
```

---

## Task Breakdown

### T1 — Adicionar `python-louvain` NÃO; usar `networkx>=3.1` nativo

**What:** Verificar que `networkx>=3.1` está em [pyproject.toml](../../../pyproject.toml) (necessário para `nx.community.louvain_communities`). Se ausente ou desatualizado, atualizar. Não adicionar `python-louvain` (design decidiu pelo nativo).
**Where:** [pyproject.toml](../../../pyproject.toml)
**Depends on:** None
**Reuses:** Lockfile existente
**Requirement:** R3.3, R5.3

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `networkx>=3.1` presente em `dependencies` (ou já estava)
- [ ] `python -c "from networkx.community import louvain_communities"` sem erro
- [ ] `torch_geometric` e `torch` versões fixadas (verificar)

**Verify:**
```bash
python -c "import networkx as nx; from networkx.algorithms.community import louvain_communities; print(nx.__version__)"
# Esperado: >= 3.1
```

**Commit:** `chore(deps): garantir networkx>=3.1 para Louvain nativo (phase 1)`

---

### T2 — Implementar `graph/temporal.py` (calendário + máscara)

**What:** Criar módulo com `week_index(date) -> int` (ISO week → offset linear `(year-2017)*52 + (week-1)`, validar range `[0, 260]`) e `mask_until(hetero, week_t) -> HeteroData` (clone raso com `edge_index/edge_attr` filtrados; aceita ambos os layouts: `first_seen_week` em `edge_attr[:, -1]` ou tensor separado `.first_seen_week`).
**Where:** [src/music_diffusion_gnn/graph/temporal.py](../../../src/music_diffusion_gnn/graph/temporal.py) (novo)
**Depends on:** T1
**Reuses:** `datetime.date.isocalendar`, `torch_geometric.data.HeteroData`
**Requirement:** R2.1, R2.2, R2.3

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `week_index("2017-01-02")` retorna 0; `week_index("2021-12-27")` retorna ≤ 260
- [ ] `week_index` levanta `ValueError` para datas fora de 2017-W1..2021-W52
- [ ] `mask_until(g, 260)` retorna grafo com mesmo número de arestas que `g`
- [ ] `mask_until(g, 0)` filtra todas exceto `has_genre` (que tem `first_seen_week=0`)
- [ ] Não faz cópia profunda de `g[ntype].x` (shared reference; checar com `id()`)
- [ ] Aceita layout misto (algumas arestas com `edge_attr`, outras com `.first_seen_week`)

**Verify:**
```bash
python -c "
from datetime import date
from music_diffusion_gnn.graph.temporal import week_index
assert week_index(date(2017,1,2)) == 0
assert week_index(date(2021,12,27)) <= 260
print('OK')
"
```

**Commit:** `feat(graph): adicionar temporal helpers (week_index, mask_until)`

---

### T3 — Implementar `nodes.build_music_nodes` [P]

**What:** Função que monta `x_music` shape `(N_m, 15)` float32 + map `song_id → idx`. Universo = `song_id` únicos no Top200 BR 2017-2021 (esperado ~6.469). Colunas: 9 acústicas + popularity + explicit + song_type + total_streams + dias_no_chart + `acoustic_missing`. Imputação mediana **antes** do z-score. Determinístico (sorted ids).
**Where:** [src/music_diffusion_gnn/graph/nodes.py](../../../src/music_diffusion_gnn/graph/nodes.py) (novo)
**Depends on:** T2
**Reuses:** `load_charts`, `load_songs` em [src/music_diffusion_gnn/data/loaders.py](../../../src/music_diffusion_gnn/data/loaders.py); `timeseries.parquet` do Phase 0
**Requirement:** R0.1, R0.4, R0.5

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `build_music_nodes(charts_df, songs_df, timeseries_df) -> tuple[Tensor, dict[str,int]]`
- [ ] `x_music.shape == (N_m, 15)` com `N_m` em `[6459, 6479]`
- [ ] Zero NaN: `assert not torch.isnan(x_music).any()`
- [ ] Coluna 15 (`acoustic_missing`) é binária `{0, 1}`
- [ ] Map é dict ordenado, keys = `sorted(song_ids)`
- [ ] Idempotente: 2 chamadas retornam mesmo tensor (bit-exact)

**Verify:**
```bash
pytest tests/test_phase1_nodes.py::test_music_shape -xvs
```

**Commit:** `feat(graph): construir music nodes com features acústicas + flag missing`

---

### T4 — Implementar `nodes.build_artist_nodes` [P]

**What:** Função que monta `x_artist` shape `(N_a, 4)` + map `artist_id → idx`. Universo = artistas com ≥1 música no `music_id_map` (esperado ~1.701). Colunas: `num_hits`, `num_collab_hits`, `years_on_charts`, `n_genres` — todas z-score.
**Where:** [src/music_diffusion_gnn/graph/nodes.py](../../../src/music_diffusion_gnn/graph/nodes.py)
**Depends on:** T2
**Reuses:** `load_artists` (campo `genres_list` já parseado); `load_charts` para cruzar com `music_id_map`
**Requirement:** R0.2

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `build_artist_nodes(artists_df, music_id_map, charts_df) -> tuple[Tensor, dict[str,int]]`
- [ ] `x_artist.shape == (N_a, 4)` com `N_a` em `[1696, 1706]`
- [ ] Zero NaN
- [ ] `n_genres` = `len(genres_list)` (sem dedup interno)
- [ ] Determinístico

**Verify:**
```bash
pytest tests/test_phase1_nodes.py::test_artist_shape -xvs
```

**Commit:** `feat(graph): construir artist nodes com features de carreira`

---

### T5 — Implementar `nodes.build_genre_nodes` [P]

**What:** Função que monta `x_genre = torch.empty(N_g, 32).normal_(0, 0.1)` + map `genre_name → idx`. Universo = união de `genres_list` dos artistas já filtrados (esperado ~530). Seed `torch.manual_seed(0)` antes da init para determinismo.
**Where:** [src/music_diffusion_gnn/graph/nodes.py](../../../src/music_diffusion_gnn/graph/nodes.py)
**Depends on:** T2
**Reuses:** `load_artists`
**Requirement:** R0.3

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `build_genre_nodes(artists_df) -> tuple[Tensor, dict[str,int]]`
- [ ] `x_genre.shape == (N_g, 32)` com `N_g` em `[520, 540]`
- [ ] 2 chamadas com mesma seed = mesmo tensor (bit-exact)
- [ ] Map ordenado alfabeticamente

**Verify:**
```bash
pytest tests/test_phase1_nodes.py::test_genre_determinism -xvs
```

**Commit:** `feat(graph): construir genre nodes com embedding aprendido (init aleatório)`

---

### T6 — Implementar `edges.build_performs` [P]

**What:** `(artist, performs, music)` direcionada. Atributos `[role, position_in_list, first_seen_week]`. Parse de `artists_id` de `br-hit_songs-*.csv` (lista ordenada: posição 0 = main, demais = feat). `first_seen_week = week_index(min(date) da música no chart)`.
**Where:** [src/music_diffusion_gnn/graph/edges.py](../../../src/music_diffusion_gnn/graph/edges.py) (novo)
**Depends on:** T2
**Reuses:** `load_songs`, `load_charts`, `week_index` de [temporal.py](../../../src/music_diffusion_gnn/graph/temporal.py)
**Requirement:** R1.1, R1.5

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `build_performs(charts_df, songs_df, music_id_map, artist_id_map) -> dict[str, Tensor]`
- [ ] Retorna `{"edge_index": LongTensor(2, E), "edge_attr": Tensor(E, 3)}`
- [ ] `edge_attr[:, 0]` ∈ `{0, 1}` (role)
- [ ] `edge_attr[:, 2]` ∈ `[0, 260]` (first_seen_week)
- [ ] Skip silencioso de artistas/músicas ausentes nos maps (com warning agregado no fim)
- [ ] Determinístico

**Verify:**
```bash
pytest tests/test_phase1_edges.py::test_performs_shape -xvs
```

**Commit:** `feat(graph): construir arestas (artist, performs, music)`

---

### T7 — Implementar `edges.build_has_genre` [P]

**What:** `(artist, has_genre, genre)` direcionada (ToUndirected adiciona reverso depois). Sem features além de `first_seen_week=0` em tensor separado. Fonte: `genres_list` por artista.
**Where:** [src/music_diffusion_gnn/graph/edges.py](../../../src/music_diffusion_gnn/graph/edges.py)
**Depends on:** T2
**Reuses:** `load_artists`
**Requirement:** R1.2, R1.5

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `build_has_genre(artists_df, artist_id_map, genre_id_map) -> dict[str, Tensor]`
- [ ] Retorna `{"edge_index": LongTensor(2, E), "first_seen_week": LongTensor(E,)}`
- [ ] Todos `first_seen_week == 0`
- [ ] Skip silencioso de genres ausentes em `genre_id_map`

**Verify:**
```bash
pytest tests/test_phase1_edges.py::test_has_genre_shape -xvs
```

**Commit:** `feat(graph): construir arestas (artist, has_genre, genre)`

---

### T8 — Implementar `edges.build_cotrajectory` [P]

**What:** `(music, cotrajectory, music)` direcionada. **2 arestas paralelas se par coexistiu em ambos os charts.** Algoritmo otimizado: agrupar `(chart, date)` → `set[song_id]`, depois iterar pares **apenas dentro do mesmo dia** acumulando dias em `dict[(i,j,chart)] -> int`. Filtrar `days_together ≥ 7`. Direção: `i→j` se primeira data de `i` < primeira de `j` no chart; empate = ordem lexicográfica. `first_seen_week` = semana do 7º dia consecutivo de coexistência (ou semana em que atinge ≥7 dias acumulados — escolher a mais simples e documentar).
**Where:** [src/music_diffusion_gnn/graph/edges.py](../../../src/music_diffusion_gnn/graph/edges.py)
**Depends on:** T2
**Reuses:** `load_charts` (ambos charts), `week_index`
**Requirement:** R1.3, R1.5

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `build_cotrajectory(charts_df, music_id_map) -> dict[str, Tensor]`
- [ ] Retorna `{"edge_index": LongTensor(2, E), "edge_attr": Tensor(E, 4)}`
- [ ] Coluna `chart` ∈ `{0, 1}` (0=viral50, 1=top200)
- [ ] Todas as arestas com `days_together >= 7`
- [ ] Custo: `<5 min` em laptop padrão; medir e logar
- [ ] Determinístico (mesma execução = mesma ordem)

**Verify:**
```bash
pytest tests/test_phase1_edges.py::test_cotrajectory_min_days -xvs
# E manual: time python -c "from music_diffusion_gnn.graph.edges import build_cotrajectory; ..."
```

**Commit:** `feat(graph): construir arestas (music, cotrajectory, music) com 2 paralelas`

---

### T9 — Implementar `edges.build_cooccurs` [P]

**What:** `(genre, cooccurs, genre)` direcionada (ToUndirected adiciona reverso). Atributos `[weight, avg_popularity, avg_streams, first_seen_week]`. Algoritmo: iterar `year ∈ [2017..2021]`, carregar `br-genre_network-{year}.csv`, manter `first_seen_week = (year-2017)*52` para par novo; atualizar `weight/avg_popularity/avg_streams` com snapshot do ano mais recente.
**Where:** [src/music_diffusion_gnn/graph/edges.py](../../../src/music_diffusion_gnn/graph/edges.py)
**Depends on:** T2
**Reuses:** `load_genre_network(year=)` em [loaders.py](../../../src/music_diffusion_gnn/data/loaders.py)
**Requirement:** R1.4, R1.5

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `build_cooccurs(genre_id_map, year_range=(2017,2021)) -> dict[str, Tensor]`
- [ ] Retorna `{"edge_index": LongTensor(2, E), "edge_attr": Tensor(E, 4)}`
- [ ] `first_seen_week ∈ {0, 52, 104, 156, 208}` (proxies anuais)
- [ ] Sem duplicatas (par `(s,t)` aparece uma única vez por direção)
- [ ] Skip silencioso de genres ausentes em `genre_id_map`

**Verify:**
```bash
pytest tests/test_phase1_edges.py::test_cooccurs_shape -xvs
```

**Commit:** `feat(graph): construir arestas (genre, cooccurs, genre) com snapshot anual`

---

### T10 — Orquestrador `graph.build.build_hetero` + validações C1-C7

**What:** Montar `HeteroData` chamando todos os builders, aplicar `ToUndirected` **seletivo** (apenas `has_genre` e `cooccurs`), validar C1-C7 inline com `assert` (mensagens claras), persistir `hetero_full.pt` e `node_id_map.json`. Atribuir `g['music'].song_id`, `g['artist'].artist_id`, `g['genre'].genre_name` (listas de strings na ordem do index).
**Where:** [src/music_diffusion_gnn/graph/build.py](../../../src/music_diffusion_gnn/graph/build.py) (novo)
**Depends on:** T3, T4, T5, T6, T7, T8, T9
**Reuses:** Todos os builders acima; `torch_geometric.transforms.ToUndirected`
**Requirement:** R3.1, R3.2, R4 (C1-C7)

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `build_hetero(out_dir: Path = Path("data/processed/graph")) -> HeteroData`
- [ ] `out_dir` criado se ausente
- [ ] `hetero_full.pt` serializado via `torch.save`
- [ ] `node_id_map.json` contém os 3 tipos com `*_to_idx` e `idx_to_*`
- [ ] Asserts C1-C7 passam para os CSVs reais (tolerâncias do spec)
- [ ] Falha rápida com mensagem clara em caso de violação
- [ ] `ToUndirected` aplicado **apenas** a `has_genre` e `cooccurs` (`performs` e `cotrajectory` permanecem direcionadas)

**Verify:**
```bash
python -c "
from pathlib import Path
from music_diffusion_gnn.graph.build import build_hetero
g = build_hetero(Path('data/processed/graph'))
print(g)
assert Path('data/processed/graph/hetero_full.pt').exists()
assert Path('data/processed/graph/node_id_map.json').exists()
"
```

**Commit:** `feat(graph): orquestrador build_hetero com validações C1-C7 inline`

---

### T11 — `graph.stats.compute_stats` (métricas + Louvain)

**What:** Função pura que recebe `HeteroData` e retorna dict com: contagens; grau (mean/median/p95/max) por `(node_type, edge_type)`; n_componentes por projeção homogênea de cada `edge_type`; clustering coefficient médio por tipo de nó (projeção); top-10 comunidades Louvain no subgrafo `(genre, cooccurs, genre)`.
**Where:** [src/music_diffusion_gnn/graph/stats.py](../../../src/music_diffusion_gnn/graph/stats.py) (novo)
**Depends on:** T10
**Reuses:** `networkx`, `nx.algorithms.community.louvain_communities`
**Requirement:** R3.3

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `compute_stats(hetero: HeteroData) -> dict`
- [ ] Dict contém keys: `counts`, `degrees`, `components`, `clustering`, `louvain_genre`
- [ ] `louvain_genre` é lista de tuplas `(comunidade_size, [genre_names])`, top-10 por tamanho
- [ ] Roda em <60s no grafo completo

**Verify:**
```bash
python -c "
import torch
from music_diffusion_gnn.graph.stats import compute_stats
g = torch.load('data/processed/graph/hetero_full.pt', weights_only=False)
s = compute_stats(g)
print(list(s.keys()))
assert len(s['louvain_genre']) <= 10
"
```

**Commit:** `feat(graph): compute_stats com graus, componentes, clustering, Louvain`

---

### T12 — `graph.stats.render_report` (markdown) [P]

**What:** Renderizar `results/phase1/stats.md` a partir do dict de `compute_stats`. Seções: contagens (com comparação ao PLANO), tabelas de grau por edge_type, número de componentes, top-10 comunidades.
**Where:** [src/music_diffusion_gnn/graph/stats.py](../../../src/music_diffusion_gnn/graph/stats.py)
**Depends on:** T11
**Reuses:** `pathlib`, string templates
**Requirement:** R3.3

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `render_report(stats: dict, out_md: Path) -> None`
- [ ] Arquivo `results/phase1/stats.md` criado
- [ ] Contém: contagens vs plano (6.469 / 1.701 / 530), 4 tabelas de grau, comunidades Louvain
- [ ] Markdown válido (pode ser lido por qualquer renderer)

**Verify:**
```bash
ls -lh results/phase1/stats.md && head -30 results/phase1/stats.md
```

**Commit:** `feat(graph): render markdown report com stats do grafo`

---

### T13 — `graph.stats.plot_degree_distributions` (figura) [P]

**What:** Gerar `results/phase1/degree_distributions.png` com 4 painéis matplotlib (um por edge_type), eixo log-log, título por painel.
**Where:** [src/music_diffusion_gnn/graph/stats.py](../../../src/music_diffusion_gnn/graph/stats.py)
**Depends on:** T11
**Reuses:** `matplotlib.pyplot`
**Requirement:** R3.4

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Assinatura: `plot_degree_distributions(hetero: HeteroData, out_png: Path) -> None`
- [ ] PNG criado em `results/phase1/degree_distributions.png`
- [ ] 4 subplots (performs, has_genre, cotrajectory, cooccurs)
- [ ] Escala log-log; eixos rotulados; títulos por subplot
- [ ] `dpi >= 100`, `bbox_inches='tight'`

**Verify:**
```bash
ls -lh results/phase1/degree_distributions.png
python -c "from PIL import Image; im = Image.open('results/phase1/degree_distributions.png'); print(im.size)"
```

**Commit:** `feat(graph): plotar distribuição de grau por tipo de aresta`

---

### T14 — `scripts/run_phase1.py` (CLI + smoke test C8/C9)

**What:** Entrypoint que executa: (1) `build_hetero`; (2) `compute_stats + render_report + plot_degree_distributions`; (3) smoke-test `HeteroSAGE` 2 camadas hidden=128 via `to_hetero`; (4) `mask_until(g, 130)` vs `(g, 260)` — assert monotônica; (5) imprimir checklist C1-C9 colorido; exit 0/1.
**Where:** [scripts/run_phase1.py](../../../scripts/run_phase1.py) (novo)
**Depends on:** T10, T11, T12, T13
**Reuses:** Padrão de [scripts/run_phase0.py](../../../scripts/run_phase0.py)
**Requirement:** R4 (C8, C9), R5.1

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `python scripts/run_phase1.py` roda end-to-end sem erro
- [ ] Tempo total < 10 min em laptop
- [ ] Smoke-test: `out['music'].shape == (≈6469, 128)`
- [ ] C9: `e130 <= e260` (monotonicidade da máscara)
- [ ] Stdout final: lista de C1..C9 com ✓/✘
- [ ] Exit code 0 se tudo passa; 1 se qualquer falha

**Verify:**
```bash
time python scripts/run_phase1.py
echo "Exit: $?"
# Esperado: < 10 min, exit 0, todos C1-C9 verdes
```

**Commit:** `feat(scripts): run_phase1.py com smoke-test HeteroSAGE + checklist C1-C9`

---

### T15 — Testes unitários `temporal` [P]

**What:** `tests/test_phase1_temporal.py` cobrindo `week_index` (limites, erros) e `mask_until` (monotonia, layout misto, shared features).
**Where:** [tests/test_phase1_temporal.py](../../../tests/test_phase1_temporal.py) (novo)
**Depends on:** T2
**Reuses:** Estilo de [tests/test_metrics.py](../../../tests/test_metrics.py)
**Requirement:** R2.1, R2.2

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `test_week_index_bounds` — datas válidas → int em `[0, 260]`
- [ ] `test_week_index_raises_oob` — datas fora levantam `ValueError`
- [ ] `test_mask_until_monotonic` — `|E(w=130)| <= |E(w=260)|`
- [ ] `test_mask_until_shared_features` — `x` é shared reference (mesmo `id()`)
- [ ] `pytest tests/test_phase1_temporal.py` verde

**Verify:**
```bash
pytest tests/test_phase1_temporal.py -xvs
```

**Commit:** `test(graph): unit tests para temporal helpers`

---

### T16 — Testes de sanidade `build` (integração) [P]

**What:** `tests/test_phase1_build.py` — único teste de integração: chama `build_hetero(tmp_path)` no dataset real, valida C1-C7 (mesmas asserts do build, mas como pytest), checa estrutura do `node_id_map.json`.
**Where:** [tests/test_phase1_build.py](../../../tests/test_phase1_build.py) (novo)
**Depends on:** T10
**Reuses:** `load_subset` para C4
**Requirement:** R4 (C1-C7)

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `test_build_hetero_counts` — C1, C2, C3 (com tolerâncias)
- [ ] `test_build_hetero_subset_coverage` — C4 (1.981 ⊆ music)
- [ ] `test_build_hetero_no_dangling_edges` — C6
- [ ] `test_build_hetero_first_seen_week_range` — C7
- [ ] Marcado com `@pytest.mark.slow` (build leva minutos); ainda rodável com `-m slow`
- [ ] `pytest tests/test_phase1_build.py -m slow` verde

**Verify:**
```bash
pytest tests/test_phase1_build.py -m slow -xvs
```

**Commit:** `test(graph): integration tests para build_hetero (C1-C7)`

---

## Parallel Execution Map

```
Wave 1:   T1 → T2

Wave 2 (after T2):
          ├── T3 [P]  build_music_nodes
          ├── T4 [P]  build_artist_nodes
          ├── T5 [P]  build_genre_nodes
          ├── T6 [P]  build_performs
          ├── T7 [P]  build_has_genre
          ├── T8 [P]  build_cotrajectory
          └── T9 [P]  build_cooccurs

Wave 3:   T3..T9 → T10  build_hetero + C1-C7

Wave 4 (after T10):
          T11  compute_stats
              ├── T12 [P]  render_report
              └── T13 [P]  plot_degree_distributions

Wave 5:   T11+T12+T13 → T14  run_phase1.py (smoke test)

Wave 6 (paralelo, sem bloquear o caminho crítico):
          T2  → T15 [P]  unit tests temporal
          T10 → T16 [P]  integration tests build
```

---

## Granularity Check

| Task | Escopo | Status |
|------|--------|--------|
| T1 | 1 dependência verificada | ✅ |
| T2 | 2 funções, 1 arquivo | ✅ |
| T3 | 1 função, 1 arquivo | ✅ |
| T4 | 1 função, mesmo arquivo | ✅ |
| T5 | 1 função, mesmo arquivo | ✅ |
| T6 | 1 função, 1 arquivo | ✅ |
| T7 | 1 função, mesmo arquivo | ✅ |
| T8 | 1 função, mesmo arquivo (algoritmo não-trivial; documentar bem) | ✅ |
| T9 | 1 função, mesmo arquivo | ✅ |
| T10 | 1 orquestrador, 1 arquivo | ✅ |
| T11 | 1 função pura | ✅ |
| T12 | 1 função (render) | ✅ |
| T13 | 1 função (plot) | ✅ |
| T14 | 1 script CLI | ✅ |
| T15 | 1 arquivo de testes | ✅ |
| T16 | 1 arquivo de testes | ✅ |

---

## Tools per task (resumo)

- **MCPs:** nenhum estritamente necessário. Stack é local (pandas/torch/PyG/networkx/matplotlib).
- **Skills:** nenhuma do skill set instalado mapeia diretamente. Trabalho é PyG puro.
- **Padrão Phase 0:** seguir estilo de `scripts/run_phase0.py` e `src/music_diffusion_gnn/baselines/*` (módulos pequenos, funções puras, asserts no orquestrador).

---

## Tips

- **`first_seen_week` precisa estar em TODA aresta** — sem isso, Phase 2 não consegue mascarar. Validação C7 captura, mas é mais barato pegar no builder.
- **Determinismo:** sempre `sorted(...)` em listas de IDs e `torch.manual_seed(0)` antes de qualquer `torch.empty().normal_()` em T5.
- **`loaders.py` caminhos defasados:** passar `path=` explícito em T3-T9; **não** refatorar agora (dívida registrada em STATE).
- **T8 (cotrajectory) é o gargalo computacional.** Se passar de 5 min, agrupar por `(chart, week)` em vez de `(chart, day)` antes de iterar pares.
- **Smoke-test em T14 não treina.** Só faz forward — se shape bate, C8 passou.
- **Commits atômicos:** 1 task = 1 commit. Mensagens no formato `feat(graph): ...` / `test(graph): ...` / `feat(scripts): ...`.
- **PR único** referenciando spec/design/tasks (R4 do spec).
