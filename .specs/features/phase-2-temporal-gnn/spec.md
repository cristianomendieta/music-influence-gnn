# Phase 2 — Temporal GNN heterogêneo

**Status:** tasks
**Janela:** semanas 3–6 (2026-05-30 → 2026-06-20)
**Depende de:** Phase 1 (`hetero_full.pt`, `mask_until`, `week_index`, `node_id_map.json`) + Phase 0 (`timeseries.parquet`)
**Bloqueia:** Phase 3 (avaliação dupla) e Phase 4 (escrita)

## Goal

Treinar um GNN temporal heterogêneo que prediz a série de popularidade
`y(t) ∈ [0, 0.5]` (mesma definição do Phase 0: rank → MA-7d → min-max → floor 0,001),
em dois regimes — **virality** (chart viral50) e **success** (chart top200) —
sob **dois objetivos** (forecasting 1-passo e fit retroativo da curva),
sem leakage temporal, com pipeline reprodutível e melhor modelo selecionado em
validação superando o baseline de persistência ingênua.

A comparação rigorosa contra SIR / wave-based e a predição genuína em k>0 são
da **Phase 3**; aqui entregamos o modelo treinado e os artefatos prontos para ela.

## Out of scope

| Item | Razão |
|---|---|
| Comparação 1-pra-1 com a Tabela do paper / Mann-Whitney vs SIR | Phase 3 (avaliação) |
| Predição genuína em horizontes k ∈ {1,7,14,30 dias} | Phase 3 Modo 2 |
| Análise interpretativa (atenção por aresta, análogos β/γ/R₀) | Phase 3 |
| Plano B (HGT / Transformer / TGN) | só se a base falhar; deferido |
| Re-treino do SIR/wave-based no regime preditivo | Phase 3 |
| Materializar snapshots semanais em disco | Phase 1 já decidiu: máscara em runtime |

---

## Requirements

### R0 — Dados, alvo e janelas

- **R0.1** Alvo: `y(t) ∈ [0, 0.5]` por tripla `(song_id, chart, week)`, lido de
  `data/processed/timeseries.parquet` (long-format do Phase 0). Dois regimes:
  `chart=viral50` → *virality*; `chart=top200` → *success*.
- **R0.2** Entrada do modelo: sequência de embeddings da música nas semanas
  `[t-W .. t-1]`, cada um produzido pelo encoder espacial sobre o snapshot do grafo.
- **R0.3** **Sem leakage:** o snapshot da semana `s` usa `mask_until(hetero, s)` —
  nenhuma aresta/feature com `first_seen_week > s` participa da predição de `y(s)`.
- **R0.4** Universo de treino: as músicas do subset viral∩hit (**1.981**, de
  `subset_ids.json["viral_intersect_hit"]`) que possuem série `y`. Tratamento de
  semanas fora do chart (`y=0`/ausência) e de janelas no início da série (`t < W`)
  é resolvido em design.md (R-open).

### R1 — Arquitetura (base do ROADMAP)

- **R1.1** Encoder espacial: **HeteroGraphSAGE** configurável (`layers ∈ {2,3}`,
  `hidden ∈ {64,128}`) → embedding por música por semana.
- **R1.2** Encoder temporal: **GRU** (1 camada, `hidden`) sobre a janela de `W`
  embeddings → estado final.
- **R1.3** Head: **MLP** → escalar com ativação que garante saída em `[0, 0.5]`
  (e.g. `0.5·sigmoid`).
- **R1.4** Ordem de grandeza **~200K parâmetros**; treina em **CPU/laptop em horas**.

### R2 — Dois objetivos de predição (decisão: *Ambos* — ver context.md)

- **R2.1** **Forecasting 1-passo** (prepara Phase 3 Modo 2): prediz `y(t)`
  usando exclusivamente dados `≤ t-1` (janela + snapshot mascarado em `t-1`).
- **R2.2** **Fit retroativo da curva** (prepara Phase 3 Modo 1): reconstrói a
  trajetória `y(t)` ao longo do período observado de cada música, análogo ao
  ajuste do SIR, para comparação 1-pra-1 posterior.
- **R2.3** Os dois objetivos **compartilham o mesmo encoder**; diferem apenas no
  protocolo de dados/avaliação. Ambos produzem `ŷ(t) ∈ [0, 0.5]`.

### R3 — Splits temporais

- **R3.1** Split **por data** (não por música): treino **2017-01 → 2020-06**,
  validação **2020-07 → 2020-12**, teste **2021-01 → 2021-12**.
- **R3.2** O **teste não é tocado** na seleção de modelo — reservado; números finais
  de teste pertencem à Phase 3.
- **R3.3** A construção de janelas que cruzam fronteiras de split usa apenas
  semanas do split corrente como alvo (a história pode vir de antes, sem alvo vazado).

### R4 — Treino e seleção de hiperparâmetros

- **R4.1** Perda **MSE** sobre `y`. Otimizador **Adam** com **weight decay**,
  **dropout** e **early stopping** no val (mitiga overfitting — risco do ROADMAP).
- **R4.2** **Grid pequeno do ROADMAP:** `W ∈ {4,8,12}`, `hidden ∈ {64,128}`,
  `layers ∈ {2,3}`, `lr ∈ {1e-3, 5e-4}`. Seleção da melhor config por **val MSE**.
- **R4.3** Seeds fixas; cada config registra métricas de treino/val.

### R5 — Baseline e critério de conclusão

- **R5.1** Baseline de **persistência ingênua** `ŷ(t) = y(t-1)` computado no val,
  para ambos os regimes.
- **R5.2** **Critério central:** a melhor config do GNN **supera a persistência no
  val MSE em ambos os regimes** (virality e success). A comparação rigorosa vs SIR
  é deferida à Phase 3.

### R6 — Saídas persistidas

- **R6.1** `results/phase2/best_model.pt` — checkpoint da melhor config (state_dict + config).
- **R6.2** `results/phase2/grid_results.parquet` — tabela do grid (config × val/train MSE × params × tempo).
- **R6.3** `results/phase2/val_predictions.parquet` — `ŷ` vs `y` no val (por regime/objetivo) para a Phase 3.
- **R6.4** `results/phase2/training_curves.png` — curvas de loss treino/val da melhor config.
- **R6.5** `results/phase2/summary.md` — relatório: melhor config, val MSE GNN vs persistência
  (ambos regimes/objetivos), nº de parâmetros, tempo de treino, decisões e desvios.

### R7 — Reprodutibilidade

- **R7.1** Pipeline regenerável com `python scripts/run_phase2.py`.
- **R7.2** Toda aleatoriedade controlada por seed configurável.
- **R7.3** Versões de torch/torch_geometric fixadas em `pyproject.toml` (já garantido na Phase 1).

---

## Acceptance criteria

A fase só conclui se TODOS passarem:

| # | Critério | Tolerância |
|---|---|---|
| C1 | `run_phase2.py` roda end-to-end sem erro em CPU | < algumas horas |
| C2 | Modelo respeita `R0.3` (sem leakage): teste unitário confirma que `y(s)` não vê aresta com `first_seen_week > s` | 0 violação |
| C3 | Saída `ŷ(t) ∈ [0, 0.5]` para todas as predições | 0 violação |
| C4 | Contagem de parâmetros da melhor config ∈ [50K, 500K] | ordem de ~200K |
| C5 | Grid completo executado (todas as combinações) com `grid_results.parquet` salvo | 0 faltando |
| C6 | **GNN supera persistência no val MSE — regime virality** | melhora > 0 |
| C7 | **GNN supera persistência no val MSE — regime success** | melhora > 0 |
| C8 | Ambos objetivos (forecasting e retroativo) produzem predições no val salvas em `val_predictions.parquet` | presentes |
| C9 | `results/phase2/summary.md` existe com tabela GNN vs persistência | presente |

---

## Acceptance test

1. `python scripts/run_phase2.py` roda end-to-end sem erro.
2. C1–C9 todos verdes; `summary.md` mostra GNN < persistência no val (MSE) nos dois regimes.
3. Artefatos R6.1–R6.5 presentes em `results/phase2/`.
4. PR único e atômico referenciando spec/design/tasks.

---

## Open questions (a resolver no design.md)

- **OQ1** Semanas fora do chart: alvo `y=0`/floor, ou treinar só na janela ativa da música?
- **OQ2** Custo do encoder espacial: recomputar embeddings GNN por ~260 semanas é caro
  (260 forwards × músicas). Cachear embeddings por semana? Subamostrar semanas-alvo?
- **OQ3** Padding de janela para `t < W` (início da série): zero-pad ou descartar a amostra.
- **OQ4** Definição precisa do *fit retroativo*: reconstrução com janela de valores reais
  (teacher forcing) vs geração livre; o que mantém a comparação justa com o SIR.
- **OQ5** Subsampling de arestas no minibatch (risco de overfitting do ROADMAP) — ligar agora ou deferir.
- **OQ6** Granularidade de batch: pares `(song, week)` por minibatch; como amostrar e embaralhar sem quebrar a causalidade temporal.

---

## Traceability

- Phase 2 spec ↔ ROADMAP.md linhas 49–73 (arquitetura, splits, grid, Plano B).
- Alvo `y ∈ [0,0.5]` ↔ Phase 0 design.md linha 16 (`rank → MA-7d → min-max [0,0.5] → floor`).
- R2 (dois objetivos) ↔ ROADMAP Phase 3 (Modo 1 retroativo / Modo 2 genuíno) + context.md.
- R3 splits ↔ ROADMAP linhas 64–67.
- R4.2 grid ↔ ROADMAP linha 69.
- Riscos (overfitting, leakage co-trajetória) ↔ ROADMAP linhas 108–115.
