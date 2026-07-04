# Phase 3 — Avaliação dupla (GNN vs SIR)

**Status:** designed (2026-06-28) — ver `design.md` (OQ1–OQ6 resolvidas)
**Janela:** semanas 6–8 (2026-06-28 → 2026-07-11) — **prazo apertado: ~2 semanas, Phase 4 ainda vem depois**
**Depende de:** Phase 2 (melhor config da grid v2, `results/phase2_experimentos_v2/`) + Phase 0 (SIR — `baselines/sir.py`, fits **a regenerar**)
**Bloqueia:** Phase 4 (escrita — todos os números finais saem daqui)

## Goal

Comparar de forma justa o Temporal GNN heterogêneo (Phase 2) contra o baseline
populacional **SIR clássico** (Phase 0), em **dois regimes** (virality/viral50,
success/top200) e **dois modos** de avaliação:

- **Modo 1 — Fit retroativo:** reconstrução da trajetória observada (análogo ao
  ajuste do SIR), comparação 1-pra-1 com a Tabela/Fig. 3 do paper.
- **Modo 2 — Predição genuína (extensão original):** forecasting causal em
  horizontes `k ∈ {1,2,4} semanas`, refazendo o SIR no mesmo regime preditivo.

Entregar também a **análise qualitativa** (Figs. 8 e 9 do paper) e, como P2, a
**análise interpretativa** (ablation por tipo de aresta, importância de features,
análogos populacionais β/γ/R₀). A saída é o conjunto de números, tabelas e figuras
que alimentam diretamente a Phase 4.

## Decisões fixadas neste specify (ver `context.md`)

| # | Decisão |
|---|---|
| D1 | **Escopo P1 = Quantitativo + Qualitativo.** Modo 1 + Modo 2 vs SIR **e** Figs 8/9 são must-ship. Análise interpretativa é **P2**. |
| D2 | **Granularidade = semanal.** Head-to-head em `y_week`; SIR regenerado e agregado para semanal. Os números diários do Phase 0 ficam como âncora de reprodução-do-paper. |
| D3 | **Horizontes Modo 2 = k ∈ {1,2,4} semanas** (≈ 7/14/30 dias do paper) via **rollout recursivo**. Critério ajustado para **≥2 de 3** horizontes. |
| D4 | **Wave-based dropado.** Comparação só vs SIR; ausência declarada como limitação. Critério "resultado forte = bater wave-based" sai de cena. |

## Out of scope

| Item | Razão |
|---|---|
| Comparação vs **wave-based** | Descartado na Phase 0 (custo proibitivo, 2026-05-12); declarar como limitação (D4) |
| Re-treino do grid do GNN | Phase 2 concluída; Phase 3 consome a melhor config existente |
| Plano B (HGT/Transformer/TGN) | Não acionado; causa-raiz da v1 foi feature, resolvida na R1 |
| Predição zero-shot p/ músicas fora do chart | Out of scope do projeto (PROJECT.md) |
| Métrica diária como head-to-head primário | D2 fixou semanal; diário fica como robustez (P3) |

---

## Insumos e dependências (R0 — regeneração obrigatória)

> ⚠️ **Achado de disco (2026-06-28):** `results/*` e `data/processed/subset_ids.json`
> são **gitignored**. `results/phase0/` (com `sir_params.parquet`) **não existe nesta
> máquina** e o `subset_ids.json` também não. Ambos são **regeneráveis**. As predições
> do GNN em disco (`results/phase2_experimentos_v2/predictions.parquet`) são da config
> fraca **W4_h64_l2_lr1e-03** ("run único"), **não** da melhor config.

- **R0.1** **GNN — melhor config.** Avaliar a melhor config da grid v2
  **`W12_h128_l3_lr5e-04`** (val_mse combinado 0.000749). Pinar o checkpoint correto
  (`best_model.pt` vs `grid_best_model.pt` em `results/phase2_experimentos_v2/`) e
  **regenerar as predições** nessa config — não reusar o `predictions.parquet` da W4.
- **R0.2** **SIR — regenerar fits.** Rodar `scripts/run_phase0.py` (ou re-fit enxuto
  via `src/music_diffusion_gnn/baselines/sir.py`) para produzir, por `(song_id, chart)`:
  parâmetros (β, γ, N, I₀) e a **trajetória diária ajustada**. Convergência foi 100%
  na Phase 0, então o re-fit do SIR é viável (foi o wave-based que era caro).
- **R0.3** **Subset.** Regenerar `data/processed/subset_ids.json` via
  `build_subset` (`viral_intersect_hit`, ~1.981 músicas) — universo de comparação.
- **R0.4** **Alvo semanal.** `y_week` = média diária por ISO-week (mesma definição da
  Phase 2). O SIR é agregado para semanal a partir da curva diária regenerada (média por
  ISO-week), garantindo `y_week` e `ŷ_week^SIR` no mesmo eixo temporal por música.

---

## Requirements

### R1 — Modo 1: Fit retroativo (reconstrução de trajetória) — **P1**

- **R1.1** **GNN (rollout livre):** partindo de uma janela-seed de `W` semanas reais,
  reconstruir a trajetória **realimentando as próprias predições** (a popularidade
  defasada `pop_bank` passa a usar valores **preditos**), análogo ao SIR *fit-then-simulate*.
  Resolve a injustiça do teacher-forcing (ver OQ1).
- **R1.2** **SIR:** curva ajustada (R0.2) agregada para semanal (R0.4).
- **R1.3** **Métrica por música:** RMSE semanal por `(song_id, chart)`; agregados
  **média ± IC 95% (bootstrap)** por regime (virality, success).
- **R1.4** **Fig. 3 replicada:** boxplot da distribuição de RMSE por música — GNN vs SIR,
  nos dois regimes.
- **R1.5** **Teste estatístico pareado:** **Wilcoxon signed-rank** (correto p/ pareado
  por música) GNN vs SIR por regime + **Mann-Whitney** reportado para alinhar com o paper.
  Reportar p-values.
- **R1.6** **Recorte de hits longos:** subgrupo com duração > 90 dias no chart
  (≈ > 13 semanas); RMSE GNN vs SIR separado (alvo PROJECT.md "forte": ≥30% de redução).

### R2 — Modo 2: Predição genuína (forecasting causal) — **P1**

- **R2.1** **GNN (rollout recursivo):** prever `y_week(w+k)` para `k ∈ {1,2,4}` usando
  exclusivamente dados `≤ w`; além do passo 1, a feature de pop defasada usa o **valor
  predito** do passo anterior. Sem leakage temporal.
- **R2.2** **SIR causal (refazer no regime preditivo):** re-ajustar o SIR **só com dados
  `≤ w`** e simular `k` semanas à frente, para cada origem de avaliação.
- **R2.3** **Baseline persistência multi-step** `ŷ(w+k) = y(w)`.
- **R2.4** **Métricas por `k` e regime:** RMSE e **acerto direcional** (sinal de `Δy`).
  Score-CRPS é **P3/deferido** (o GNN é determinístico; CRPS exigiria ensemble/MC-dropout).
- **R2.5** **Critério:** GNN supera o SIR em **≥2 de 3** horizontes, por regime.

### R3 — Análise qualitativa (Figs. 8 e 9) — **P1**

- **R3.1** Replicar Figs. 8/9 do paper para os casos **"Shallow", "Batom de Cereja",
  "Água Nos Zói", "abcdefu"**: curva real vs GNN vs SIR ao longo do tempo.
- **R3.2** Se algum caso não estiver no subset/período (2017–2021), **substituir por caso
  comparável e documentar** a troca.

### R4 — Análise interpretativa — **P2**

- **R4.1** **Ablation por tipo de aresta** (substitui "atenção": o encoder é HeteroGraphSAGE,
  **sem pesos de atenção**): remover cada tipo (artista→música, gênero↔gênero, co-trajetória),
  re-medir RMSE no val, reportar Δ por tipo.
- **R4.2** **Importância de grupos de features:** permutação de features **acústicas**
  vs **metadados** vs **pop defasada**; reportar Δ RMSE (ablation do risco "acústicas dominam").
- **R4.3** **Análogos populacionais β/γ/R₀** extraídos das trajetórias do GNN (taxas de
  subida/descida) para comparação interpretativa com os parâmetros do SIR.

### R5 — Saídas persistidas — **P1/P2**

- **R5.1** `results/phase3/mode1_per_song.parquet` — RMSE por `(song,chart,modelo)` (Modo 1).
- **R5.2** `results/phase3/mode2_horizons.parquet` — RMSE/acerto direcional por `(k,regime,modelo)`.
- **R5.3** `results/phase3/fig3_boxplot.png`, `results/phase3/figs_8_9_casos.png`.
- **R5.4** `results/phase3/interpretability.parquet` (P2) — Δ RMSE por ablation/feature.
- **R5.5** `results/phase3/summary.md` — tabela-mãe: GNN vs SIR (Modo 1 + Modo 2, dois
  regimes), p-values, recorte de hits longos, decisões e desvios.

### R6 — Reprodutibilidade

- **R6.1** Pipeline regenerável com `python scripts/run_phase3.py` (inclui R0.1–R0.4).
- **R6.2** Toda aleatoriedade (bootstrap, permutação, seeds do SIR) controlada por seed.

---

## Acceptance criteria

A fase só conclui se TODOS os P1 passarem (C9–C11 são P2):

| # | Critério | Prioridade | Tolerância |
|---|---|---|---|
| C1 | `run_phase3.py` roda end-to-end (regenera SIR + subset + predições GNN) sem erro | P1 | — |
| C2 | SIR re-gerado e GNN (melhor config W12) avaliados no **mesmo eixo semanal** e mesmo subset | P1 | 0 desalinhamento |
| C3 | **Modo 1:** boxplot Fig.3 + RMSE médio ±IC95% + p-value (Wilcoxon) por regime, salvos | P1 | presentes |
| C4 | **Modo 1, critério mínimo:** GNN < SIR em RMSE de **success** com **p < 0,01** | P1 | bate o alvo |
| C5 | **Modo 2:** RMSE + acerto direcional por `k ∈ {1,2,4}` e regime, salvos | P1 | presentes |
| C6 | **Modo 2, critério mínimo:** GNN < SIR em **≥2 de 3** horizontes | P1 | bate o alvo |
| C7 | Recorte de hits longos (>90 dias) reportado (GNN vs SIR) | P1 | presente |
| C8 | Figs. 8/9 replicadas p/ os 4 casos (ou substitutos documentados) | P1 | 4 casos |
| C9 | Ablation por tipo de aresta (Δ RMSE) reportado | P2 | presente |
| C10 | Importância acústicas vs metadados vs pop defasada reportada | P2 | presente |
| C11 | Análogos β/γ/R₀ extraídos do GNN | P2 | presente |
| C12 | `summary.md` com a tabela-mãe GNN vs SIR | P1 | presente |

> **Nota sobre os critérios de sucesso do PROJECT.md:** C4 = "mínimo aceitável" (Modo 1);
> C6 = "mínimo aceitável" (Modo 2). "Resultado forte vs wave-based" foi **removido** (D4).
> "Redução ≥30% em hits longos" (R1.6) permanece como alvo de **resultado forte**, não gate.

---

## Acceptance test

1. `python scripts/run_phase3.py` roda e produz `results/phase3/`.
2. C1–C8 + C12 (P1) verdes; `summary.md` mostra a tabela-mãe GNN vs SIR.
3. Se Modo 1/Modo 2 **não** baterem o SIR: registrar como "resultado limitado mas
   publicável" (PROJECT.md) — reposicionamento, não falha de pipeline.
4. PR único e atômico referenciando spec/design/tasks.

---

## Edge cases

- WHEN um dos 4 casos qualitativos não existe no subset/período THEN substituir por caso
  comparável (mesma faixa de duração/regime) e documentar (R3.2).
- WHEN o SIR causal (R2.2) não converge para uma origem (poucos dados `≤ w`) THEN excluir
  aquela origem da média e registrar a contagem de exclusões.
- WHEN o rollout do GNN (R1.1/R2.1) diverge (clamp saturando em 0 ou 0,5) THEN reportar a
  taxa de saturação como diagnóstico, não mascarar.
- WHEN a granularidade semanal suaviza demais a estrutura multi-onda (favorecendo o SIR em
  hits longos) THEN o recorte de hits longos (R1.6) expõe o efeito — declarar no summary.

---

## Open questions (resolver no design.md)

- **OQ1 — Fairness do Modo 1 (CRÍTICA).** A cabeça residual do GNN é `ŷ = clamp(y_prev+Δ)`.
  Sob *teacher forcing*, `y_prev` é o valor real → a reconstrução vira quase-persistência e
  **vence o SIR trivialmente** (o SIR não recebe `y(w-1)`). Resolução recomendada: Modo 1 =
  **rollout livre** a partir de janela-seed (análogo a "fit-then-simulate" do SIR); o
  teacher-forced de 1 passo é o Modo 2 `k=1`. Definir tamanho da seed e como o resíduo
  ancora durante o rollout (sem `y_real` defasado).
- **OQ2 — Checkpoint exato.** `best_model.pt` vs `grid_best_model.pt`; o ckpt do Colab é da
  W12? Re-treinar a melhor config localmente (CPU, horas) ou confiar no ckpt baixado.
- **OQ3 — SIR causal.** Janela mínima de dados `≤ w` para fit estável (precisa do pico
  observado?); quais semanas servem de **origem** de avaliação (todas? grid?); custo de
  re-fit × origens × `k`.
- **OQ4 — Estatística.** Wilcoxon (pareado) como primário + Mann-Whitney (alinhar paper);
  método de IC 95% (bootstrap por música).
- **OQ5 — "Hit longo".** Definição operacional de >90 dias no diário ≡ semanas no chart.
- **OQ6 — CRPS.** Incluir via ensemble/MC-dropout (P3) ou deferir totalmente.

---

## Requirement traceability

| ID | Requisito | Story/Modo | Prioridade | Status |
|---|---|---|---|---|
| EVAL-01 | Regenerar SIR + subset + predições GNN (W12) | R0 | P1 | Pending |
| EVAL-02 | Modo 1 GNN rollout livre vs SIR (semanal) | R1.1–R1.2 | P1 | Pending |
| EVAL-03 | RMSE/música ±IC95% + Fig.3 + Wilcoxon | R1.3–R1.5 | P1 | Pending |
| EVAL-04 | Recorte de hits longos | R1.6 | P1 | Pending |
| EVAL-05 | Modo 2 GNN rollout k∈{1,2,4} vs SIR causal | R2.1–R2.3 | P1 | Pending |
| EVAL-06 | RMSE + acerto direcional por k/regime | R2.4–R2.5 | P1 | Pending |
| EVAL-07 | Figs. 8/9 — 4 casos | R3 | P1 | Pending |
| EVAL-08 | Ablation por tipo de aresta | R4.1 | P2 | Pending |
| EVAL-09 | Importância de grupos de features | R4.2 | P2 | Pending |
| EVAL-10 | Análogos β/γ/R₀ | R4.3 | P2 | Pending |
| EVAL-11 | Artefatos persistidos + summary.md | R5 | P1 | Pending |
| EVAL-12 | `run_phase3.py` reprodutível | R6 | P1 | Pending |

**Coverage:** 12 requisitos; **12/12 mapeados** a tasks (`tasks.md`, 12 tasks em 5 waves).

---

## Success criteria (da fase)

- [ ] **Mínimo (PROJECT.md):** C4 (Modo 1, success, p<0,01) **e** C6 (Modo 2, ≥2/3 horizontes).
- [ ] **Forte:** ≥30% de redução de RMSE em hits longos (R1.6).
- [ ] **Piso publicável:** se não bater o SIR, ganho num subgrupo (gênero/duração) +
      reposicionamento "competitivo com modelos populacionais, com sinais relacionais".
- [ ] Todos os P1 (C1–C8, C12) verdes; P2 (C9–C11) entregues se o prazo permitir.

---

## Traceability (cross-doc)

- Phase 3 ↔ ROADMAP.md linhas 76–93 (avaliação dupla, Modos 1/2, qualitativo, interpretativo).
- Critérios mínimos ↔ PROJECT.md linhas 43–48 (success RMSE Modo 1 p<0,01; ≥2 horizontes Modo 2).
- D4 (wave-based) ↔ STATE.md 2026-05-12 (wave-based descartado).
- D2 (granularidade) ↔ Phase 2 design.md (decisão T-gran deferida à Phase 3) + STATE 2026-05-30.
- R0.1 (melhor config) ↔ STATE 2026-06-28 (W12_h128_l3; polish: avaliar na melhor config).
- R0.2 (regenerar SIR) ↔ achado de disco: `results/` gitignored, `phase0/` ausente.
