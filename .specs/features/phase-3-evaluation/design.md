# Phase 3 — Avaliação dupla (GNN vs SIR) — Design

**Spec:** `.specs/features/phase-3-evaluation/spec.md`
**Context:** `.specs/features/phase-3-evaluation/context.md`
**Status:** Draft
**Data:** 2026-06-28

> Este design resolve as 6 open questions do spec (OQ1–OQ6) com base em
> **sondagens reais do código e dos dados** (registradas abaixo), não em suposições.
> A novidade load-bearing é o **rollout livre/recursivo** do GNN (OQ1); o resto é
> orquestração que **reusa** Phase 0 (SIR) e Phase 2 (modelo, dataset, métricas).

---

## 0. Achados de sondagem (fundam as decisões)

Verificado nesta sessão (2026-06-28) contra o repo e os artefatos em disco:

| # | Achado | Impacto no design |
|---|---|---|
| S1 | **`grid_best_model.pt`** é um *wrapper* `{config_str,W,hidden,layers,lr,dropout,val_mse,state_dict}` e contém a melhor config **W12_h128_l3_lr5e-04** (hidden=128, layers=3, val_mse 0.000749). O `state_dict` **inclui o buffer `pop_bank`**. | **OQ2 resolvida:** carregar `grid_best_model.pt` (não `best_model.pt`). |
| S2 | **`best_model.pt`** é um *state_dict cru* da config fraca **W4_h64_l2** (GRU 192×64 → hidden=64; 2 conv layers). | **Não usar** em Phase 3 (confirma o alerta do spec). |
| S3 | `timeseries.parquet` (versionado, 4,44M linhas, 1981 músicas) é **100% denso**: há uma linha por dia-calendário no span de cada `(song,chart)`, com `y` **pisado em 0.001** nos dias fora do chart (`rank_score==0`). 92,7% das linhas estão ≤0.001. | **OQ5 resolvida** e exige cuidado no Modo 1 (ver §1.5). "Dias no chart" = `rank_score>0`, **não** o span denso. |
| S4 | **Dias genuínos no chart** (`rank_score>0`), por regime: viral50 mediana **14d**, **>90d = 77 músicas (4%)**; top200 mediana **55d**, **>90d = 798 (40%)**. | **OQ5:** o recorte ">90d" é um **minoritário informativo** (não 97%). Alinha com o paper (SIR fraco em hits longos, comuns no success, raros no virality). |
| S5 | `MusicDiffusionGNN.pop_bank` é um **buffer mutável**; `encode_weeks(g,[w])` lê `self.pop_bank[w]` (injeção de feature) e `predict` lê `y_prev=pop_bank[w-1,song,chart]` (âncora de persistência). | **OQ1:** rollout = **mutar `pop_bank` em cópia de trabalho** e re-encodar. Mecânica já existe; basta orquestrar. |
| S6 | Phase 2 já encoda **uma vez por semana distinta** e reusa o bank por grupo de `target_week` (`_iter_batches`/`encode_weeks`). Um passe full-graph sobre as 261 semanas ≈ custo de ~1 época (minutos em CPU). | **OQ1/perf:** rollout global sincronizado ≈ **um passe** (encode cada semana 1×), não O(músicas×span×W). |
| S7 | `sir_params.parquet` guarda só `(beta,gamma,R0,rmse,converged,n_iter)` — **não** a curva. `_sir_curve(t,β,γ,I0)` reconstrói a curva diária; `I0=y[0]`. | **R0.2/Modo 1 SIR:** reconstituir a curva diária do fit e **agregar por ISO-week** (mesma regra de `aggregate_weekly`). |

---

## 1. Resolução das Open Questions

### OQ1 — Fairness do Modo 1 → **rollout livre global sincronizado** (CRÍTICA)

**Problema (do spec):** sob *teacher forcing*, `predict` lê `y_prev = pop_bank[w-1]` = valor **real** ⇒ a cabeça residual `ŷ=clamp(y_prev+Δ)` vira quase-persistência e bate o SIR **trivialmente** (o SIR não recebe `y(w-1)`).

**Resolução:** Modo 1 = **rollout livre** (análogo direto ao SIR *fit-then-simulate*). A partir de uma **janela-seed** das `seed_weeks` primeiras semanas reais de cada `(song,chart)`, a trajetória é reconstruída **realimentando as próprias predições**: tanto a feature de pop injetada (`pop_bank[w]`) quanto a âncora `y_prev=pop_bank[w-1]` passam a usar **valores preditos** fora da seed. Assim `y_prev` deixa de ser o valor real → sem persistência trivial.

**`seed_weeks = W` (=12 na melhor config).** Justificativa: (a) cobre a primeira janela de look-back sem padding; (b) S4/S6 mostram que 98% das músicas têm ≥12 semanas observadas ⇒ a seed **não** sacrifica cobertura; (c) a seed é uma fração pequena do span mediano (~166 semanas) ⇒ o rollout continua sendo uma reconstrução longa e genuinamente difícil. Parametrizável (`--seed-weeks`). Edge: span ≤ W ⇒ `seed = max(1, span-1)` (logado); span < 2 ⇒ excluída (contada).

**Algoritmo (global sincronizado — uma passada, todas as músicas juntas):**

```
work_bank = pop_bank.clone()            # buffer de trabalho mutável
model.pop_bank = work_bank              # reassocia o buffer (restaura no fim)
Zcache: dict[week → Z_music] = {}       # encoding lazy, 1×/semana

para w = 1 .. 260 (ordem crescente):
    janela = [w-W .. w-1]               # todas < w ⇒ já finalizadas
    para j na janela, se j∉Zcache e j≥0:
        Zcache[j] = model.encode_weeks(g,[j])[j]     # 1 forward full-graph
    amostras_w = todos (song,chart) com target_week=w no span ativo
    ŷ = model.predict({j:Zcache[j] for j in janela}, amostras_w)   # vetorizado
    para cada (song,chart) FORA da seed:  work_bank[w,song,chart] = ŷ
    # músicas ainda na seete: work_bank[w] mantém o valor real
```

Custo ≈ 261 `encode_weeks` (S6) + predicts baratos ⇒ **minutos em CPU**. A região
avaliada por música é `w ∈ [first_seen+seed .. last_seen]`.

> **Nota de significado (registrar no paper):** o critério mínimo **C4** só é
> informativo porque o Modo 1 é rollout livre. O teacher-forced de 1 passo é
> exatamente o **Modo 2 `k=1`**.

### OQ2 — Checkpoint exato → **`grid_best_model.pt`** (resolvida por S1/S2)

Carregar `results/phase2_experimentos_v2/grid_best_model.pt`:
```python
ck = torch.load(path, map_location=device, weights_only=False)
model = MusicDiffusionGNN(g.metadata(), n_genre=g["genre"].num_nodes,
                          hidden=ck["hidden"], layers=ck["layers"],
                          dropout=ck["dropout"], pop_bank=pop_bank_regen)
sd = ck["state_dict"]; sd.pop("pop_bank", None)      # usar pop_bank regenerado
missing, unexpected = model.load_state_dict(sd, strict=False)
assert set(missing) <= {"pop_bank"} and not unexpected
```
`pop_bank_regen` vem de `build_pop_bank(weekly_df, …)` (R0). **Guard de reprodutibilidade:** assert `allclose(pop_bank_regen, ck_pop_bank)` onde ambos definidos; divergência ⇒ warn (drift de subset/timeseries). **Sem re-treino local** (a melhor config já está pinada; treinar em CPU custaria horas — STATE 2026-06-23).

### OQ3 — SIR causal (Modo 2) → **refit `≤w` + simulate, restrito ao test span**

- **Origens de avaliação:** semanas `w ∈ [TEST_START_WEEK(208) .. 260]` (regime de
  forecasting genuíno, alinhado ao held-out da Phase 2). `origin_stride=1` (parametrizável; subir se custo apertar).
- **Por origem `(song,chart,w)`:** refit `fit_sir` na série **diária ≤ fim-da-semana-w**; simular `k∈{1,2,4}` semanas à frente via `_sir_curve`; agregar por ISO-week → `ŷ^SIR_week(w+k)`.
- **Janela mínima p/ fit estável:** exigir `≥ min_hist_weeks` (default **4**) semanas com `rank_score>0` em `≤w`. Não exige o pico observado (o SIR ajusta β,γ com o que houver; convergência foi 100% na Phase 0).
- **Não-convergência** (`converged==False`) ⇒ excluir a origem da média e **contar exclusões** (edge case do spec).
- **Custo:** `fit_sir` ~ms; paralelizar com o padrão `baselines/parallel.fit_all` (joblib loky). #fits ≈ (origens on-chart no test span) — viável em minutos.

### OQ4 — Estatística → **Wilcoxon pareado (primário) + Mann-Whitney + bootstrap IC95%**

- **Primário (pareado por música):** `scipy.stats.wilcoxon(rmse_gnn, rmse_sir)` por regime (pares = mesma `(song,chart)`). Correto para o pareamento.
- **Secundário (alinhar paper):** `mann_whitney_pairwise` (já em `metrics.py`).
- **IC 95%:** **bootstrap por música** (reamostrar `(song)` com reposição, B=**10000**, intervalo percentílico) sobre a média de RMSE por modelo e sobre a **diferença** GNN−SIR. Seed controlada.
- Reportar p-values dos dois testes + IC da diferença. Empates/`zero_method`: usar `wilcoxon(..., zero_method="wilcox")` (default) e reportar `n` de pares.

### OQ5 — "Hit longo" → **>90 dias com `rank_score>0`** (resolvida por S3/S4)

`days_on_chart(song,chart) = #dias com rank_score>0` (computado do diário **antes** da
agregação semanal). **Long = >90 dias.** Resultado: viral50 **77 (4%)**, top200 **798 (40%)**.
RMSE GNN vs SIR **dentro** do subgrupo long, por regime; alvo **≥30% de redução** (resultado
forte, **não** gate — spec). Por ser 4% no viral50, reportar com ressalva de `n` pequeno.

> **Armadilha evitada:** definir "long" pelo span denso (S3) ou por `y>floor` capturaria
> ~97% das músicas — métrica quase idêntica à global, inútil. O recorte por `rank_score>0`
> isola o tail SIR-fraco que o paper visa.

### OQ6 — CRPS → **deferido (P3)**

O GNN é determinístico; CRPS exigiria ensemble/MC-dropout. **Fora do escopo da Phase 3**
(confirma R2.4). Caminho futuro registrado: ativar `dropout` em `eval` (MC-dropout, T=30
amostras) → `properscoring.crps_ensemble`. Não bloqueia nenhum gate.

### Decisão transversal — região de avaliação do Modo 1 (consequência de S3)

RMSE semanal do Modo 1 sobre o **span observado completo** `[first_seen+seed .. last_seen]`
(inclui a cauda no floor 0.001) como **primário** — é a mesma região em que o SIR foi
ajustado na Phase 0 (consistência/reprodução) e ambos os modelos enfrentam o mesmo alvo.
Como **robustez** (bundle com hits longos), reportar também RMSE só nas **semanas on-chart**
(`rank_score>0`), para evidenciar que eventual vitória do GNN **não** é só "prever o floor".

---

## 2. Architecture Overview

```mermaid
flowchart TD
    subgraph R0["R0 — Regeneração (idempotente)"]
        A[build_subset] --> B[subset_ids.json]
        C[timeseries.parquet versionado] --> D[aggregate_weekly → weekly_df]
        C --> E[fit_all + fit_sir → sir_params.parquet]
        D --> F[build_pop_bank → pop_bank_regen]
        G[grid_best_model.pt W12] --> H[MusicDiffusionGNN carregado]
        F --> H
    end

    subgraph M1["Modo 1 — Fit retroativo (semanal)"]
        H --> I[gnn_rollout_free → ŷ_week por música]
        E --> J[sir_weekly_from_fit → ŷ^SIR_week]
        I --> K[mode1_per_song.parquet RMSE/música]
        J --> K
        K --> L[stats: Wilcoxon+MW+bootstrap]
        K --> M[fig3_boxplot.png GNN vs SIR]
        K --> N[recorte hits longos >90d]
    end

    subgraph M2["Modo 2 — Predição genuína (k=1,2,4)"]
        H --> O[gnn_rollout_recursive ≤w]
        C --> P[sir_causal_refit ≤w + simulate]
        D --> Q[persistence multi-step ŷ=y w]
        O --> R[mode2_horizons.parquet RMSE+dir.acc.]
        P --> R
        Q --> R
    end

    subgraph QUAL["Qualitativo + Interpretativo"]
        I --> S[figs_8_9_casos.png]
        J --> S
        H --> T[(P2) ablation aresta / perm. features / β,γ,R₀]
    end

    K --> U[summary.md — tabela-mãe + checklist C1–C12]
    R --> U
    N --> U
    T --> U
```

Orquestrado por `scripts/run_phase3.py` (idempotente, no estilo `run_phase0`/`run_phase2`).

---

## 3. Code Reuse Analysis

### Componentes existentes a alavancar

| Componente | Local | Como usar |
|---|---|---|
| `MusicDiffusionGNN` (`encode_weeks`, `predict`, buffer `pop_bank`) | `models/diffusion_gnn.py` | Núcleo do rollout (M1/M2): mutar `pop_bank` de trabalho + re-encodar. |
| `aggregate_weekly`, `temporal_split`, `build_samples`, `build_pop_bank`, `Sample`, `TRAIN_END_WEEK/TEST_START_WEEK` | `training/dataset.py` | Alvo semanal, splits, `pop_bank`, origens do Modo 2. |
| `_iter_batches`, `_distinct_window_weeks`, `_eval_mse` (padrão de encode 1×/semana) | `training/trainer.py` | Padrão de batching por semana p/ o rollout global. |
| `fit_sir`, `SIRFit`, `_sir_curve` | `baselines/sir.py` | Refit causal (M2) e reconstrução da curva diária (M1). |
| `fit_all` (joblib loky) | `baselines/parallel.py` | Paralelizar refits do SIR causal e o fit Phase 0. |
| `persistence_predict` / `_bulk` | `models/baselines.py` | Estender p/ persistência **multi-step** (R2.3). |
| `rmse`, `mann_whitney_pairwise`, `summarize_rmse` | `evaluation/metrics.py` | Métricas + alinhamento c/ paper; estender com Wilcoxon/bootstrap. |
| `make_boxplot` (padrão Fig.3) | `evaluation/report.py` | Base p/ o boxplot **GNN vs SIR** (2 modelos lado a lado). |
| `build_subset`/`load_subset` | `data/subset.py` | R0.3. |
| `run_phase0.main` | `scripts/run_phase0.py` | R0.2 (regenerar SIR + subset) — chamar como subrotina idempotente. |

### Pontos de integração

| Sistema | Método de integração |
|---|---|
| Grafo `hetero_full.pt` + `node_id_map.json` | `torch.load` / passado a `encode_weeks` e `build_pop_bank` (mesmas chamadas da Phase 2). |
| `results/phase0/sir_params.parquet` | Produzido por `run_phase0`; lido p/ reconstruir curva (M1). |
| `results/phase2_experimentos_v2/grid_best_model.pt` | Pinned (S1); única fonte do modelo. |
| `results/phase3/` | Diretório-alvo dos artefatos (R5), gitignored (criar `.gitkeep`). |

---

## 4. Componentes e Interfaces

> Novo subpacote **`src/music_diffusion_gnn/evaluation/`** (já existe; adicionar módulos).
> Mantém o padrão Phase 0/2 (funções puras + um orquestrador em `scripts/`).

### `evaluation/rollout.py` — rollout do GNN (núcleo, OQ1)
- **Purpose:** reconstrução/predição realimentando predições no `pop_bank`.
- **Interfaces:**
  - `gnn_rollout_free(model, g, weekly_df, *, W, seed_weeks, device) -> pd.DataFrame`
    — colunas `[song_id, chart, week, y_true, y_pred]` (Modo 1, global sincronizado).
  - `gnn_rollout_recursive(model, g, weekly_df, origins, *, W, ks=(1,2,4), device) -> pd.DataFrame`
    — colunas `[song_id, chart, origin_week, k, y_true, y_pred]` (Modo 2).
  - `_saturation_rate(preds) -> float` — fração em 0 ou 0.5 (edge case: diagnóstico, não mascarar).
- **Dependencies:** `MusicDiffusionGNN`, `Sample`, grafo. **Reuses:** `encode_weeks`/`predict`, padrão `_distinct_window_weeks`.

### `evaluation/sir_eval.py` — SIR nos dois modos
- **Purpose:** curva semanal do SIR (M1) e refit causal+simulate (M2).
- **Interfaces:**
  - `sir_weekly_from_fit(daily_y, fit: SIRFit) -> pd.Series` (indexada por `week`) — reconstrói via `_sir_curve(t,β,γ,I0)` e agrega por ISO-week (regra de `aggregate_weekly`).
  - `sir_causal_forecast(daily_y_upto_w, k_weeks, *, min_hist_weeks=4) -> dict[k→float] | None` — refit `≤w` + simulate; `None` se não convergir.
  - `run_sir_mode1(ts_df, sir_params_df) -> pd.DataFrame` / `run_sir_mode2(ts_df, origins, ks) -> pd.DataFrame` (paralelizado via `fit_all`-style).
- **Reuses:** `fit_sir`, `_sir_curve`, `parallel.fit_all`.

### `evaluation/stats.py` — testes pareados + IC
- **Interfaces:**
  - `wilcoxon_signed_rank(rmse_a, rmse_b) -> {statistic, p_value, n}`
  - `bootstrap_ci_mean(values, *, B=10000, seed) -> (lo, hi, mean)`
  - `bootstrap_ci_diff(rmse_a, rmse_b, *, B=10000, seed) -> (lo, hi, mean_diff)`
  - `directional_accuracy(y_origin, y_true_k, y_pred_k) -> float` (R2.4)
- **Reuses:** `scipy.stats`; co-localizar com `mann_whitney_pairwise`.

### `evaluation/longhits.py` — recorte de duração (OQ5)
- **Interface:** `days_on_chart(ts_df) -> pd.Series` (index `(song_id,chart)`, `=Σ[rank_score>0]`); `long_hit_mask(ts_df, *, threshold_days=90) -> set[(song_id,chart)]`.

### `evaluation/figures.py` — figuras do paper
- **Interfaces:**
  - `fig3_boxplot_gnn_vs_sir(mode1_df, out_path)` — boxplot RMSE/música, 2 regimes × 2 modelos (estende `make_boxplot`).
  - `figs_8_9_cases(mode1_df, cases, out_path)` — curva real vs GNN vs SIR p/ os 4 casos nomeados; substituição documentada se ausente (R3.2).

### `evaluation/interpretability.py` — **P2** (R4)
- **Interfaces:**
  - `edge_type_ablation(model, g, val_samples) -> pd.DataFrame` (Δ RMSE removendo cada `edge_type`).
  - `feature_group_permutation(model, g, val_samples, groups) -> pd.DataFrame` (acústicas/metadados/pop defasada).
  - `population_analogs(mode1_df) -> pd.DataFrame` (β/γ/R₀ análogos das taxas de subida/descida das trajetórias do GNN).

### `scripts/run_phase3.py` — orquestrador (R6)
- **Purpose:** end-to-end idempotente: **R0** (chama `run_phase0.main` + carrega modelo) → **M1** → **M2** → **figuras** → **summary + checklist C1–C12**.
- **Flags:** `--seed 42`, `--smoke` (poucas músicas/origens), `--skip-interpret` (pula P2), `--seed-weeks`, `--origin-stride`, `--device`.
- **Reuses:** estilo `_banner/_check/_elapsed` do `run_phase2`.

---

## 5. Data Models (esquemas persistidos — R5)

```python
# R5.1 results/phase3/mode1_per_song.parquet
{ "song_id": str, "chart": str, "model": str,   # "gnn" | "sir"
  "rmse": float, "rmse_onchart": float,          # span completo / só rank>0
  "days_on_chart": int, "is_long_hit": bool, "n_weeks_eval": int,
  "saturation_rate": float }                      # só gnn (diagnóstico)

# R5.2 results/phase3/mode2_horizons.parquet
{ "song_id": str, "chart": str, "model": str,    # "gnn"|"sir"|"persist"
  "origin_week": int, "k": int,                  # k ∈ {1,2,4}
  "y_true": float, "y_pred": float, "converged": bool }
# agregação (k,regime,model): RMSE + directional_accuracy derivados em summary

# R5.4 (P2) results/phase3/interpretability.parquet
{ "analysis": str, "component": str, "delta_rmse": float, "regime": str }
```
- **mode1_per_song**: granularidade 1 linha por `(song,chart,model)`; pares para Wilcoxon/bootstrap.
- **mode2_horizons**: granularidade 1 linha por `(song,chart,origin,k,model)`; permite RMSE e acerto direcional por `(k,regime)`.
- **Figuras:** `fig3_boxplot.png`, `figs_8_9_casos.png` (R5.3). **summary.md** (R5.5): tabela-mãe + checklist.

---

## 6. Error Handling Strategy

| Cenário | Tratamento | Impacto |
|---|---|---|
| SIR causal não converge p/ origem `(song,w)` (M2) | excluir origem da média; **contar exclusões** no summary | métrica honesta; transparência |
| Rollout do GNN diverge (clamp satura em 0/0.5) | reportar `saturation_rate` por regime como diagnóstico | não mascarar (edge case spec) |
| Caso qualitativo ausente no subset/período | substituir por caso comparável (mesma faixa duração/regime) e **documentar** (R3.2) | Fig 8/9 ainda entregue |
| `pop_bank_regen ≠ checkpoint pop_bank` | `warn` + seguir com regenerado; abortar só se shape divergir | guard de reprodutibilidade |
| Música com span < 2 (sem janela de rollout) | excluir + contar | cobertura declarada |
| `results/phase0/` ausente (S3 do spec) | `run_phase0.main()` regenera (idempotente) | R0 satisfeita |
| Semanal suaviza estrutura multi-onda (favorece SIR em hits longos) | exposto pelo recorte de hits longos + RMSE on-chart | declarado no summary (edge case spec) |

---

## 7. Tech Decisions (não óbvias)

| Decisão | Escolha | Racional |
|---|---|---|
| Checkpoint | `grid_best_model.pt` (W12) | S1/S2: único que é a melhor config; `best_model.pt` é a W4 fraca. |
| Rollout Modo 1 | **global sincronizado** (encode 1×/semana, todas as músicas) | S6: ~um passe (minutos) vs O(músicas×span×W) inviável. |
| `seed_weeks` | `W` (=12) | S4/S6: cobre look-back sem padding, sem perder cobertura (98% têm ≥12 sem). |
| Origens do Modo 2 | test span (sem 208–260) | regime de forecasting genuíno, alinhado ao held-out da Phase 2; bound de custo. |
| "Hit longo" | `>90d` com `rank_score>0` | S3/S4: span denso/floor capturaria 97%; `rank>0` isola o tail SIR-fraco (4%/40%). |
| Região RMSE M1 | span completo (primário) + on-chart (robustez) | consistência c/ fit do SIR (Phase 0) + blindar contra "só prever o floor". |
| Teste primário | Wilcoxon pareado | pareamento por música; Mann-Whitney como secundário p/ alinhar paper. |
| CRPS | deferido (P3) | GNN determinístico; exige ensemble/MC-dropout — fora do prazo. |
| Re-treino local | **não** | melhor config pinada; CPU custaria horas (STATE 2026-06-23). |

---

## 8. Orçamento de desempenho (estimativa)

| Etapa | Custo estimado | Observação |
|---|---|---|
| R0 SIR refit (Phase 0) | ~minutos | joblib; convergência 100% histórica. |
| M1 GNN rollout global | ~poucos minutos (CPU) | ~261 `encode_weeks` (S6). |
| M1 SIR weekly | segundos | reconstrução analítica da curva. |
| M2 GNN recursivo | ~minutos | ~53 origens × ≤4 passos, encodes ≤w compartilhados. |
| M2 SIR causal | ~minutos | fits ~ms paralelizados; bound pelo test span. |
| Bootstrap (B=10k) | segundos | vetorizado por regime. |
| Interpretabilidade (P2) | ~minutos | só se `--skip-interpret` não setado. |

GPU local incompatível (STATE 2026-06-23) ⇒ **CPU**. Sem treino, só inferência/fit: viável no prazo.

---

## 9. Itens em aberto p/ Tasks (não bloqueiam o design)

- Confirmar a lista exata de `edge_type`s do grafo p/ a ablation R4.1 (ler `g.edge_types` no início da task P2).
- Definir os 4 `song_id` dos casos nomeados (lookup por nome no MGD+; preparar substitutos por faixa de duração — R3.2).
- `min_hist_weeks` do SIR causal (default 4) pode ser calibrado na execução se exclusões > X%.

---

## Traceabilidade design → requisitos

| Requisito (spec) | Coberto por |
|---|---|
| R0.1–R0.4 | §1 OQ2, §3 (run_phase0), `build_pop_bank`, `aggregate_weekly` |
| R1.1 (rollout livre) | §1 OQ1, `rollout.gnn_rollout_free` |
| R1.2 (SIR semanal) | `sir_eval.sir_weekly_from_fit` |
| R1.3–R1.5 (RMSE±IC, Fig3, Wilcoxon/MW) | `stats.py`, `figures.fig3_*` |
| R1.6 (hits longos) | §1 OQ5, `longhits.py` |
| R2.1–R2.3 (M2 GNN/SIR/persist) | `rollout.gnn_rollout_recursive`, `sir_causal_forecast`, persistência multi-step |
| R2.4–R2.5 (métricas/critério k) | `stats.directional_accuracy`, summary |
| R3 (Figs 8/9) | `figures.figs_8_9_cases` |
| R4 (interpretativo, P2) | `interpretability.py` |
| R5 (artefatos) | §5 esquemas + `run_phase3` |
| R6 (reprodutível/seed) | `run_phase3` idempotente, seeds em bootstrap/SIR/rollout |
```
