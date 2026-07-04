# ROADMAP — 10 semanas até BraSNAM 2026

Janela operacional: **2026-05-02 → 2026-07-11**.
Cada fase abaixo vira uma feature em `.specs/features/<slug>/` quando entrar em planejamento.

## Visão geral

```
W1───W2───W3───W4───W5───W6───W7───W8───W9───W10
[ Phase 0 ]
       [ Phase 1 ]
            [    Phase 2    ]
                            [ Phase 3 ]
                                        [ Phase 4 ]
```

## Phase 0 — Reprodução dos baselines (semanas 1–2)

**Slug:** `phase-0-baselines`
**Status:** ✅ completed (2026-05-12)
**Bloqueia:** todas as fases seguintes (sem reprodução, comparação posterior é contaminada).

**Resultados obtidos:**
- SIR RMSE virality: **0,0289** (target ≈0,028 ± 10%) ✅
- SIR RMSE success: **0,0471** (target ≈0,052 ± 10%) ✅
- Mann-Whitney p-value: **1,61e-39** ✅
- Subset: **1.981 músicas** (paper: 1.977; δ de 4 por diferença de período) ✅
- Convergência SIR: **100%** ✅
- Wave-based: **descartado** por decisão do pesquisador (ver STATE.md 2026-05-12)

Artefatos em `results/phase0/`: `summary.md`, `sir_params.parquet`, `boxplot_fig3.png`.

## Phase 1 — Construção do grafo heterogêneo (semanas 2–3)

**Slug:** `phase-1-hetero-graph`
**Status:** specifying (2026-05-17)
**Depende de:** Phase 0 (mesmas séries pré-processadas).

**Schema de nós:** Música (6.469), Artista (1.701), Gênero (530).
**Tipos de aresta:** artista→música (interpreta), artista—gênero (pertence_a),
música→música (co-trajetória, ≥7 dias juntos no chart), gênero—gênero (co-ocorrência MGD+).
**Temporal:** snapshots semanais.

**Ferramentas:** `networkx` para análise; `torch_geometric.data.HeteroData` para treino.

**Saída:** estatísticas exploratórias (distribuição de grau por tipo, componentes,
clustering, comunidades por gênero) + objeto `HeteroData` serializado.

## Phase 2 — Modelagem com Temporal GNN heterogêneo (semanas 3–6)

**Slug:** `phase-2-temporal-gnn`
**Status:** ✅ completed (R1, 2026-06-28) — v1 perdia p/ persistência; R1 (popularidade defasada
como feature de nó + cabeça residual) **bate persistência nas 24 configs** (grid v2, Colab T4).
Melhor: W12_h128_l3_lr5e-04, val_mse combinado 0.000749. Artefatos em `results/phase2_experimentos_v2/`.
**Depende de:** Phase 1.

**Arquitetura base:**
```
[snapshot semana t] → HeteroGraphSAGE (2 camadas, hidden=128)
                    → embedding por música em t (128-d)
[seq embeddings t-W..t-1] → GRU (hidden=128, 1 camada)
                          → MLP → rank_score(t) ∈ [0, 0.5]
```
~200K parâmetros. Treina em CPU/laptop em horas.

**Splits temporais:**
- Treino: 2017-01 → 2020-06 (3,5 anos).
- Validação: 2020-07 → 2020-12.
- Teste: 2021-01 → 2021-12.

**Grid pequeno:** W ∈ {4, 8, 12}, hidden ∈ {64, 128}, layers ∈ {2, 3}, lr ∈ {1e-3, 5e-4}.

**Plano B (se base não funcionar):** HGT no lugar de HeteroSAGE; Transformer no
lugar de GRU; TGN puro.

## Phase 3 — Avaliação dupla (semanas 6–8)

**Slug:** `phase-3-evaluation`
**Status:** designed (2026-06-28) — spec + context + design escritos. OQ1–OQ6 resolvidas:
Modo 1 = **rollout livre global** (`seed_weeks=W`, encode 1×/semana); checkpoint = **`grid_best_model.pt`**
(W12; `best_model.pt` é a W4 fraca); SIR causal refit `≤w` no test span; **Wilcoxon** pareado +
bootstrap IC95%; **hit longo = >90d `rank_score>0`** (viral50 4%, top200 40%); CRPS deferido.
⚠️ SIR e `subset_ids.json` **não estão em disco** (`results/` gitignored) → R0 regenera via `run_phase0.py`.
**Depende de:** Phase 2 (melhor config W12_h128_l3, `results/phase2_experimentos_v2/`) e Phase 0 (SIR a regenerar).

**Modo 1 — Fit retroativo:** comparação 1-pra-1 com Tabela/Fig. 3 do paper, em granularidade
**semanal**. Métricas: RMSE médio ± IC 95% (bootstrap), boxplot (Fig. 3), Wilcoxon pareado
(+ Mann-Whitney p/ alinhar paper) vs SIR. GNN = **rollout livre** (justiça vs teacher forcing, OQ1).

**Modo 2 — Predição genuína (extensão original):** `y_week(w+k)` usando dados ≤ w, via rollout
recursivo. Refazer SIR causal no mesmo regime. Métricas: RMSE em **k ∈ {1, 2, 4} semanas**,
acerto direcional; score-CRPS deferido (P3).

**Análise qualitativa:** replicar Figs. 8 e 9 do paper com casos "Shallow",
"Batom de Cereja", "Água Nos Zói", "abcdefu".

**Análise interpretativa:** atenção por tipo de aresta, importância de features
acústicas vs metadados, análogos populacionais (β, γ, R₀) extraídos do GNN.

## Phase 4 — Escrita e submissão (semanas 8–10)

**Slug:** `phase-4-paper`
**Status:** pending
**Depende de:** Phase 3 (todos os números finais).

**Estrutura SBC (8–12 páginas):** Intro · Trabalhos relacionados · Dados ·
Metodologia · Avaliação · Discussão · Conclusão.

**Citações imprescindíveis (não estavam no plano anterior):**
Wave-based ASONAM 2025; Causalidade IEEE Access 2025; WebSci 2024 (viral songs);
HGT (Hu 2020); TGN (Rossi 2020).

## Riscos cross-fase

| Risco | Onde mitiga |
|---|---|
| Overfitting do GNN (1.179 músicas) | Phase 2: dropout + weight decay + early stopping + subsampling de arestas |
| Vazamento via co-trajetória | Phase 1: aresta só conta com data ≤ t; Phase 3: validar splits |
| Wave-based difícil de reproduzir | Phase 0: contatar autor; fallback mistura de Gaussianas |
| GNN não bate wave-based | Phase 3 + Phase 4: reposicionar como "resultado limitado" |
| Features acústicas dominarem o sinal estrutural | Phase 3: ablation com/sem acústicas |
| Diferença de 2,5 meses no período | Phase 4: declarar como limitação |

## Status atual

- ✅ **Phase 0 concluída** (2026-05-12) — todos os critérios de aceitação passaram.
- Repo limpo: `scripts/exploratory/` para diagnósticos; `scripts/run_phase0.py` único entry point.
- **Próximo passo:** design da Phase 1 via `/tlc-spec-driven design phase-1-hetero-graph`
  (resolver 5 open questions: features de gênero, arestas paralelas vs união, imputação acústica,
  direção `has_genre`, embedding inicial de gênero).
