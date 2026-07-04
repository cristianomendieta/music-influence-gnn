# Phase 3 — Tasks (Avaliação dupla GNN vs SIR)

**Design:** [`design.md`](design.md) · **Spec:** [`spec.md`](spec.md) · **Context:** [`context.md`](context.md)
**Status:** Draft

> 12 tasks atômicas em 5 waves. Cada task = 1 commit. PR único final referenciando spec/design/tasks.
> Stack numpy/scipy/torch/matplotlib (igual Phase 0/2) — **sem MCPs/Skills** necessários (confirmado nas Phases 1 e 2).
> Subpacote-alvo: `src/music_diffusion_gnn/evaluation/` (já existe; `metrics.py` e `report.py` presentes — **reusar, não recriar**).

---

## Política de testes (não há `.specs/codebase/TESTING.md`)

Inferida da convenção do repo (pytest, arquivos `tests/test_phaseN_*.py`, marker `slow`, runner `.venv/bin/python -m pytest`):

| Camada criada | Tipo de teste exigido | Gate |
|---|---|---|
| Funções puras (stats, longhits, curva SIR, persistência) | **unit** co-localizado | quick |
| Rollout do GNN (inferência + **propriedade de não-leakage**) | **unit + leakage** co-localizado | quick |
| Figuras (saída visual) | **smoke** (PNG criado, não-vazio) | quick |
| Orquestrador `run_phase3.py` | **integration** via `--smoke` | full |
| Execução real end-to-end | acceptance (não-pytest) | — |

- **quick:** `.venv/bin/python -m pytest tests/test_phase3_*.py -q`
- **full:** `.venv/bin/python -m pytest -q` (suite inteira; garante zero regressão nas Phases 0–2)
- **smoke e2e:** `.venv/bin/python scripts/run_phase3.py --smoke`

> ⚠️ `.venv` aponta interpreter de path antigo: usar `.venv/bin/python` (e `-m pip`), **não** `.venv/bin/pip` (STATE.md).

---

## Execution Plan

### Wave 0 — Folhas (paralelo; sem código novo dependente)
```
T1 [P]  (model_io — loader W12 + guard)
T2 [P]  (longhits — recorte de duração)
T3 [P]  (stats — Wilcoxon/bootstrap/dir.acc.)
T4 [P]  (baselines — persistência multi-step)
```

### Wave 1 — Núcleos SIR + Rollout
```
T5 (sir_eval M1) ──→ T6 (sir_eval M2 causal)
T1 ──────────────→ T7 (rollout livre M1) ──→ T8 (rollout recursivo M2)
```

### Wave 2 — Figuras + Interpretabilidade (paralelo)
```
T5,T7 ──→ T9  (figures — Fig.3 + Figs.8/9)
T1 ──────→ T10 (interpretability — P2)
```

### Wave 3 — Orquestrador (sequencial)
```
T2,T3,T4,T6,T8,T9 (+T10 opcional) ──→ T11 (run_phase3.py)
```

### Wave 4 — Execução real (sequencial)
```
T11 ──→ T12 (rodar end-to-end + registrar STATE + veredito C1–C12)
```

---

## Parallel Execution Map
```
Wave 0:  T1 [P]   T2 [P]   T3 [P]   T4 [P]
Wave 1:  T5 ──→ T6              (sequencial; mesmo módulo sir_eval.py)
         T7 ──→ T8              (sequencial; mesmo módulo rollout.py; T7 após T1)
Wave 2:  T9 [P] (após T5,T7)    T10 [P] (após T1)
Wave 3:  T11                    (após T2,T3,T4,T6,T8,T9; T10 se sem --skip-interpret)
Wave 4:  T12                    (execução real)
```

`evaluation/` = `src/music_diffusion_gnn/evaluation/` · `models/` = `src/music_diffusion_gnn/models/` ·
testes = `tests/` (raiz, padrão Phase 1/2).

---

## Task Breakdown

### T1: `load_grid_best_model` em `evaluation/model_io.py` [P]
**What:** Carregar o checkpoint **W12** (`grid_best_model.pt`) como `MusicDiffusionGNN`, dropando o `pop_bank` do state_dict e injetando o regenerado, com guard de reprodutibilidade.
**Where:** `src/music_diffusion_gnn/evaluation/model_io.py` (novo)
**Depends on:** None
**Reuses:** `MusicDiffusionGNN` [models/diffusion_gnn.py:65](../../../src/music_diffusion_gnn/models/diffusion_gnn.py#L65); `build_pop_bank` [training/dataset.py:212](../../../src/music_diffusion_gnn/training/dataset.py#L212)
**Requirement:** EVAL-01 (R0.1), OQ2/S1/S2 (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `load_grid_best_model(ckpt_path, g, pop_bank_regen, *, device) -> MusicDiffusionGNN` constrói com `hidden/layers/dropout` lidos do wrapper (`ck["hidden"]`, etc.)
- [ ] `sd = ck["state_dict"]; sd.pop("pop_bank", None)`; `load_state_dict(sd, strict=False)` com `assert set(missing) <= {"pop_bank"} and not unexpected`
- [ ] Guard: `allclose(pop_bank_regen, ck_pop_bank)` onde ambos definidos → `warn` (não aborta) se divergir; aborta só se **shape** divergir
- [ ] Recusa explícita de `best_model.pt` (W4): docstring + checagem de que o caminho default é `grid_best_model.pt`
- [ ] Unit test: carrega o ckpt real, confere `config_str` contém `W12_h128_l3`, `count_params()` na faixa ~200K, e que `missing ⊆ {pop_bank}`

**Tests:** unit · **Gate:** quick

**Verify:**
```bash
.venv/bin/python -c "
import torch
from music_diffusion_gnn.training.dataset import aggregate_weekly, build_pop_bank
from music_diffusion_gnn.evaluation.model_io import load_grid_best_model
import pandas as pd
g = torch.load('data/processed/graph/hetero_full.pt', weights_only=False)
nmap='data/processed/graph/node_id_map.json'
w = aggregate_weekly(pd.read_parquet('data/processed/timeseries.parquet'))
pb = build_pop_bank(w, nmap, n_music=g['music'].num_nodes)
m = load_grid_best_model('results/phase2_experimentos_v2/grid_best_model.pt', g, pb, device='cpu')
print('params', m.count_params())"
```

**Commit:** `feat(phase3): load W12 grid-best checkpoint with pop_bank guard`

---

### T2: `days_on_chart` + `long_hit_mask` em `evaluation/longhits.py` [P]
**What:** Recorte de duração por `rank_score>0` (não pelo span denso) — define o subgrupo "hit longo" (>90 dias).
**Where:** `src/music_diffusion_gnn/evaluation/longhits.py` (novo)
**Depends on:** None
**Reuses:** `timeseries.parquet` (cols `song_id,chart,rank_score`)
**Requirement:** EVAL-04 (R1.6), OQ5/S3/S4 (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `days_on_chart(ts_df) -> pd.Series` index `(song_id,chart)` = `Σ[rank_score>0]` (computado do **diário**, antes de agregar)
- [ ] `long_hit_mask(ts_df, *, threshold_days=90) -> set[tuple[str,str]]`
- [ ] Unit test sintético: série com N dias `rank>0` e M dias `==0` → `days_on_chart == N`; máscara inclui só `>90`
- [ ] Comentário fixando a armadilha (span denso/floor capturaria ~97%; `rank>0` isola o tail)

**Tests:** unit · **Gate:** quick

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase3_longhits.py -q
```

**Commit:** `feat(phase3): long-hit cut by days with rank_score>0`

---

### T3: `evaluation/stats.py` — testes pareados + IC + acerto direcional [P]
**What:** Wilcoxon pareado, bootstrap IC95% (média e diferença), acerto direcional.
**Where:** `src/music_diffusion_gnn/evaluation/stats.py` (novo)
**Depends on:** None
**Reuses:** `scipy.stats`; co-localizar com `mann_whitney_pairwise` [evaluation/metrics.py:13](../../../src/music_diffusion_gnn/evaluation/metrics.py#L13)
**Requirement:** EVAL-03 (R1.3–R1.5), EVAL-06 (R2.4), OQ4 (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `wilcoxon_signed_rank(rmse_a, rmse_b) -> {statistic, p_value, n}` (`zero_method="wilcox"`, reporta `n` de pares)
- [ ] `bootstrap_ci_mean(values, *, B=10000, seed) -> (lo, hi, mean)` (percentílico, reamostragem por música)
- [ ] `bootstrap_ci_diff(rmse_a, rmse_b, *, B=10000, seed) -> (lo, hi, mean_diff)` (pareado por música)
- [ ] `directional_accuracy(y_origin, y_true_k, y_pred_k) -> float` (fração de acerto do sinal de `Δy`)
- [ ] Determinismo: mesmo `seed` ⇒ mesmo IC (unit test)
- [ ] Unit tests: Wilcoxon contra caso conhecido; bootstrap reprodutível; `directional_accuracy` em sinais controlados

**Tests:** unit · **Gate:** quick

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase3_stats.py -q
```

**Commit:** `feat(phase3): paired Wilcoxon, bootstrap CI, directional accuracy`

---

### T4: `persistence_multistep` em `models/baselines.py` [P]
**What:** Baseline de persistência multi-step `ŷ(w+k) = y(w)` para o Modo 2.
**Where:** `src/music_diffusion_gnn/models/baselines.py` (estender; já tem `persistence_predict`/`_bulk`)
**Depends on:** None
**Reuses:** padrão de [models/baselines.py:8](../../../src/music_diffusion_gnn/models/baselines.py#L8)
**Requirement:** EVAL-05 (R2.3)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `persistence_multistep(weekly_df, origins, ks) -> pd.DataFrame` colunas `[song_id, chart, origin_week, k, y_pred]` com `y_pred = y_week(origin)`
- [ ] Sem leakage (usa só `y(origin)`, nunca `> origin`)
- [ ] Unit test: `y_pred(k)` constante = `y(origin)` para todo `k`

**Tests:** unit · **Gate:** quick

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase3_baselines.py -q
```

**Commit:** `feat(phase3): multi-step persistence baseline (ŷ(w+k)=y(w))`

---

### T5: `sir_weekly_from_fit` + `run_sir_mode1` em `evaluation/sir_eval.py`
**What:** Modo 1 do SIR — reconstruir a curva **diária** do fit (params Phase 0) e agregar para semanal (mesma regra de `aggregate_weekly`).
**Where:** `src/music_diffusion_gnn/evaluation/sir_eval.py` (novo)
**Depends on:** None
**Reuses:** `_sir_curve` [baselines/sir.py:32](../../../src/music_diffusion_gnn/baselines/sir.py#L32), `SIRFit` [baselines/sir.py:15](../../../src/music_diffusion_gnn/baselines/sir.py#L15); regra de agregação de `aggregate_weekly` [training/dataset.py:38](../../../src/music_diffusion_gnn/training/dataset.py#L38)
**Requirement:** EVAL-02 (R1.2), S7 (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `sir_weekly_from_fit(daily_y, fit) -> pd.Series` indexada por `week`: `_sir_curve(t, β, γ, I0=daily_y[0])` → agrega por ISO-week (média)
- [ ] `run_sir_mode1(ts_df, sir_params_df) -> pd.DataFrame` colunas `[song_id, chart, week, y_pred_sir]` para todo o subset
- [ ] Eixo semanal idêntico ao do GNN (mesmo `week_index`) — verificado em teste
- [ ] Unit test sintético: fit conhecido (β,γ) → curva diária → semanal com nº de semanas esperado e valores ∈ [0,0.5]

**Tests:** unit · **Gate:** quick

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase3_sir_eval.py -q
```

**Commit:** `feat(phase3): SIR Mode-1 weekly curve from Phase-0 fit`

---

### T6: `sir_causal_forecast` + `run_sir_mode2` em `evaluation/sir_eval.py`
**What:** Modo 2 do SIR — refit causal **só com dados `≤w`** + simular `k∈{1,2,4}` semanas à frente, por origem.
**Where:** `src/music_diffusion_gnn/evaluation/sir_eval.py` (mesmo módulo)
**Depends on:** T5
**Reuses:** `fit_sir` [baselines/sir.py:42](../../../src/music_diffusion_gnn/baselines/sir.py#L42); `fit_all` (paralelismo joblib) [baselines/parallel.py:23](../../../src/music_diffusion_gnn/baselines/parallel.py#L23); `sir_weekly_from_fit` (T5)
**Requirement:** EVAL-05 (R2.2), OQ3 (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `sir_causal_forecast(daily_y_upto_w, k_weeks, *, min_hist_weeks=4) -> dict[k→float] | None`: refit `≤w`, simula, agrega por ISO-week; `None` se `converged==False` **ou** histórico `<min_hist_weeks` semanas com `rank>0`
- [ ] `run_sir_mode2(ts_df, origins, ks=(1,2,4)) -> pd.DataFrame` colunas `[song_id, chart, origin_week, k, y_pred_sir, converged]`; origens restritas ao **test span** (`week ≥ TEST_START_WEEK=208`)
- [ ] Não-convergência/origem-curta → linha com `converged=False` (contabilizada a jusante, não silenciada)
- [ ] Paralelizado (joblib) sem alterar resultados (determinístico)
- [ ] Unit test: série com pico claro converge e produz `k=1,2,4`; série curta (<`min_hist_weeks`) ⇒ `None`

**Tests:** unit · **Gate:** quick

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase3_sir_eval.py -q
```

**Commit:** `feat(phase3): causal SIR refit+simulate for Mode-2 horizons`

---

### T7: `gnn_rollout_free` + `_saturation_rate` em `evaluation/rollout.py`
**What:** Modo 1 do GNN — **rollout livre global sincronizado** (OQ1): a partir da janela-seed (`seed_weeks=W`), realimenta as próprias predições no `pop_bank` de trabalho.
**Where:** `src/music_diffusion_gnn/evaluation/rollout.py` (novo)
**Depends on:** T1
**Reuses:** `MusicDiffusionGNN.encode_weeks`/`predict`/buffer `pop_bank` [models/diffusion_gnn.py:95](../../../src/music_diffusion_gnn/models/diffusion_gnn.py#L95); padrão de encode 1×/semana [training/trainer.py](../../../src/music_diffusion_gnn/training/trainer.py); `Sample` [training/dataset.py:107](../../../src/music_diffusion_gnn/training/dataset.py#L107)
**Requirement:** EVAL-02 (R1.1), OQ1/S5/S6 (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `gnn_rollout_free(model, g, weekly_df, *, W, seed_weeks, device) -> pd.DataFrame` colunas `[song_id, chart, week, y_true, y_pred]`
- [ ] Algoritmo do design §1: `work_bank = pop_bank.clone()`, reassociado ao modelo (restaurado no fim); encode **1×/semana** (cache lazy); fora da seed `work_bank[w] ← ŷ` (predito), na seed mantém o real
- [ ] Edge: `span ≤ W` ⇒ `seed = max(1, span-1)` (logado); `span < 2` ⇒ excluída (contada)
- [ ] `_saturation_rate(preds) -> float` = fração em 0 ou 0.5 (diagnóstico, **não** mascarar — edge case spec)
- [ ] Smoke unit test (subset pequeno, `--device cpu`): roda sem erro; colunas corretas; nas semanas da seed `y_pred` provém do real (âncora), fora da seed `y_prev` **não** é o real

**Tests:** unit · **Gate:** quick

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase3_rollout.py -q
```

**Commit:** `feat(phase3): free synchronized GNN rollout for Mode-1 (OQ1)`

---

### T8: `gnn_rollout_recursive` em `evaluation/rollout.py`
**What:** Modo 2 do GNN — rollout recursivo por origem: prever `y_week(w+k)` usando só dados `≤w`; além do passo 1, a pop defasada usa o **valor predito** do passo anterior.
**Where:** `src/music_diffusion_gnn/evaluation/rollout.py` (mesmo módulo)
**Depends on:** T7
**Reuses:** `encode_weeks`/`predict`/`pop_bank` (idem T7)
**Requirement:** EVAL-05 (R2.1), OQ1 (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `gnn_rollout_recursive(model, g, weekly_df, origins, *, W, ks=(1,2,4), device) -> pd.DataFrame` colunas `[song_id, chart, origin_week, k, y_true, y_pred]`
- [ ] Sem leakage: `encode_weeks` só com semanas `≤ origin`; para `k≥2` a pop defasada é o `ŷ(w+k-1)` predito (nunca `y_real(>origin)`)
- [ ] Origens restritas ao test span (`≥ TEST_START_WEEK=208`)
- [ ] **Leakage test** co-localizado (assertivo, não probabilístico): para uma origem `w`, nenhuma chamada de `encode_weeks` recebe semana `>w`; `work_bank[>origin]` nunca recebe valor real durante o rollout
- [ ] Smoke unit test: `k=1,2,4` produzidos; shapes corretos

**Tests:** unit (+ leakage) · **Gate:** quick

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase3_rollout.py tests/test_phase3_leakage.py -q
```

**Commit:** `test(phase3): recursive GNN rollout for Mode-2 with no-leakage assertion`

---

### T9: `evaluation/figures.py` — Fig.3 + Figs.8/9 [P]
**What:** Boxplot RMSE/música GNN vs SIR (Fig.3, 2 regimes × 2 modelos) e curvas real vs GNN vs SIR para os 4 casos nomeados (Figs.8/9).
**Where:** `src/music_diffusion_gnn/evaluation/figures.py` (novo)
**Depends on:** T5, T7
**Reuses:** `make_boxplot` (estender p/ 2 modelos) [evaluation/report.py:23](../../../src/music_diffusion_gnn/evaluation/report.py#L23); esquema `mode1_per_song` (design §5)
**Requirement:** EVAL-03 (R1.4), EVAL-07 (R3)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `fig3_boxplot_gnn_vs_sir(mode1_df, out_path) -> Path` — boxplot 2×2 (regime × modelo); salva PNG
- [ ] `figs_8_9_cases(curves_df, cases, out_path) -> Path` — curva real/GNN/SIR p/ "Shallow", "Batom de Cereja", "Água Nos Zói", "abcdefu"
- [ ] Caso ausente no subset/período (2017–2021) ⇒ substituir por caso comparável (mesma faixa duração/regime) e **anotar a troca** no título/legenda (R3.2)
- [ ] Smoke test: chamar com DataFrame sintético gera PNG **não-vazio** (size > 0) sem erro

**Tests:** smoke · **Gate:** quick

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase3_figures.py -q
```

**Commit:** `feat(phase3): Fig.3 boxplot and Figs.8/9 case curves (GNN vs SIR)`

---

### T10: `evaluation/interpretability.py` — P2 (R4) [P]
**What:** Ablation por tipo de aresta, importância de grupos de features (permutação) e análogos populacionais β/γ/R₀.
**Where:** `src/music_diffusion_gnn/evaluation/interpretability.py` (novo)
**Depends on:** T1
**Reuses:** `MusicDiffusionGNN` (encode/predict), `g.edge_types` (lido no início — design §9), `rmse` [evaluation/metrics.py:9](../../../src/music_diffusion_gnn/evaluation/metrics.py#L9)
**Requirement:** EVAL-08/09/10 (R4.1–R4.3) — **P2**

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `edge_type_ablation(model, g, val_samples) -> pd.DataFrame` (Δ RMSE removendo cada `edge_type` do `g.edge_types`)
- [ ] `feature_group_permutation(model, g, val_samples, groups) -> pd.DataFrame` (acústicas / metadados / pop defasada; Δ RMSE)
- [ ] `population_analogs(curves_df) -> pd.DataFrame` (β/γ/R₀ análogos das taxas de subida/descida das trajetórias do GNN)
- [ ] Saída no esquema `interpretability.parquet` (design §5): `[analysis, component, delta_rmse, regime]`
- [ ] Smoke test: cada função roda em subset minúsculo e retorna DataFrame com as colunas esperadas

**Tests:** smoke · **Gate:** quick

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase3_interpretability.py -q
```

**Commit:** `feat(phase3): P2 interpretability (edge ablation, feature perm, β/γ/R₀)`

---

### T11: `scripts/run_phase3.py` — orquestrador idempotente
**What:** End-to-end: **R0** (`run_phase0.main()` + carregar modelo + `build_pop_bank`) → **M1** → **M2** → **figuras** → **summary.md + checklist C1–C12**; persiste R5.1–R5.5.
**Where:** `scripts/run_phase3.py` (novo)
**Depends on:** T2, T3, T4, T6, T8, T9 (T10 se `--skip-interpret` ausente)
**Reuses:** `run_phase0.main` [scripts/run_phase0.py:30](../../../scripts/run_phase0.py#L30); estilo `_banner/_step/_elapsed` de `run_phase0`/`run_phase2`; `aggregate_weekly`/`build_pop_bank`; todas as funções T1–T10
**Requirement:** EVAL-01/03/04/06/11/12 (R0, R5, R6)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] **R0:** `run_phase0.main()` (regenera `subset_ids.json` + `sir_params.parquet`, idempotente); `load_grid_best_model` (T1); `build_pop_bank`
- [ ] **M1:** `gnn_rollout_free` (T7) + `run_sir_mode1` (T5) → RMSE/música (span completo **e** on-chart `rank>0`) → `mode1_per_song.parquet` (R5.1) com `days_on_chart`/`is_long_hit` (T2), `saturation_rate` (T7)
- [ ] **M2:** `gnn_rollout_recursive` (T8) + `run_sir_mode2` (T6) + `persistence_multistep` (T4) → `mode2_horizons.parquet` (R5.2); RMSE + `directional_accuracy` (T3) por `(k,regime,modelo)`; contar exclusões (não-convergência)
- [ ] **Estatística:** Wilcoxon + Mann-Whitney + bootstrap IC95% da diferença (T3), por regime
- [ ] **Figuras:** `fig3_boxplot.png` + `figs_8_9_casos.png` (T9) (R5.3)
- [ ] **P2 (opcional):** se sem `--skip-interpret`, `interpretability.parquet` (R5.4) via T10
- [ ] **summary.md** (R5.5): tabela-mãe GNN vs SIR (Modo 1+2, 2 regimes), p-values, recorte hits longos, exclusões, desvios + **checklist C1–C12** com verde/vermelho; `exit 1` se algum P1 (C1–C8, C12) falhar
- [ ] `results/phase3/.gitkeep` criado (dir gitignored)
- [ ] Flags: `--seed 42`, `--smoke`, `--skip-interpret`, `--seed-weeks`, `--origin-stride`, `--device`
- [ ] **Smoke gate:** `run_phase3.py --smoke` roda end-to-end (poucas músicas/origens) e cria `results/phase3/*` sem erro

**Tests:** integration · **Gate:** full

**Verify:**
```bash
.venv/bin/python scripts/run_phase3.py --smoke && ls results/phase3/
.venv/bin/python -m pytest -q   # zero regressão Phases 0–2
```

**Commit:** `feat(phase3): run_phase3 orchestrator with R5 artifacts and C1-C12 checklist`

---

### T12: Rodar Phase 3 completo + registrar resultados
**What:** Execução real end-to-end na melhor config; conferir C1–C8/C12 (e C9–C11 se prazo); registrar números/desvios em STATE.md.
**Where:** execução + `.specs/project/STATE.md`
**Depends on:** T11
**Reuses:** —
**Requirement:** EVAL-12, acceptance test

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `python scripts/run_phase3.py` roda end-to-end sem erro (C1); tempo medido (minutos, CPU)
- [ ] C2–C8 + C12 verdes; `summary.md` mostra a tabela-mãe GNN vs SIR (Modo 1+2)
- [ ] C4 (Modo 1 success, p<0,01) e C6 (Modo 2 ≥2/3 horizontes) avaliados; se **não** baterem o SIR ⇒ registrar como "resultado limitado mas publicável" (reposicionamento, não falha de pipeline)
- [ ] Recorte de hits longos (>90d) reportado; ganho ≥30% checado como **resultado forte** (não-gate)
- [ ] Artefatos R5.1–R5.5 presentes em `results/phase3/`
- [ ] STATE.md atualizado: RMSE GNN vs SIR por modo/regime, p-values, exclusões, hits longos, P2 (se feito), desvios
- [ ] (se prazo) C9–C11 (P2) entregues

**Tests:** acceptance · **Gate:** —

**Verify:**
```bash
.venv/bin/python scripts/run_phase3.py && ls results/phase3/
```

**Commit:** `chore(phase3): record dual-evaluation results and phase outcome in STATE`

---

## Pre-Approval Validation

### Check 1 — Task Granularity

| Task | Escopo | Status |
|------|--------|--------|
| T1 model_io.load_grid_best_model | 1 função + guard | ✅ |
| T2 longhits | 2 funções coesas (1 módulo) | ✅ |
| T3 stats | 4 funções coesas (1 módulo) | ✅ |
| T4 persistence_multistep | 1 função | ✅ |
| T5 sir_eval M1 | 2 funções coesas | ✅ |
| T6 sir_eval M2 | 2 funções coesas | ✅ |
| T7 rollout livre | 1 função + helper | ✅ |
| T8 rollout recursivo | 1 função + leakage test | ✅ |
| T9 figures | 2 funções (1 módulo) | ✅ |
| T10 interpretability (P2) | 3 funções (1 módulo) | ✅ |
| T11 run_phase3 | 1 entrypoint | ✅ |
| T12 execução | run + registro | ✅ |

### Check 2 — Diagram-Definition Cross-Check

| Task | Depends On (corpo) | Diagrama mostra | Status |
|------|--------------------|-----------------|--------|
| T1 | None | folha (Wave 0) | ✅ |
| T2 | None | folha (Wave 0) | ✅ |
| T3 | None | folha (Wave 0) | ✅ |
| T4 | None | folha (Wave 0) | ✅ |
| T5 | None | T5 → T6 (origem) | ✅ |
| T6 | T5 | T5 → T6 | ✅ |
| T7 | T1 | T1 → T7 | ✅ |
| T8 | T7 | T7 → T8 | ✅ |
| T9 | T5, T7 | T5,T7 → T9 | ✅ |
| T10 | T1 | T1 → T10 | ✅ |
| T11 | T2,T3,T4,T6,T8,T9 (+T10 opc.) | …→ T11 | ✅ |
| T12 | T11 | T11 → T12 | ✅ |

Tarefas `[P]` (T1–T4 na Wave 0; T9/T10 na Wave 2) não dependem entre si. ✅

### Check 3 — Test Co-location Validation

> Sem `TESTING.md`; matriz inferida acima (seção "Política de testes").

| Task | Camada criada | Matriz exige | Task diz | Status |
|------|---------------|--------------|----------|--------|
| T1 | loader (lógica IO) | unit | unit | ✅ |
| T2 | função pura | unit | unit | ✅ |
| T3 | funções puras | unit | unit | ✅ |
| T4 | função pura | unit | unit | ✅ |
| T5 | lógica numérica | unit | unit | ✅ |
| T6 | lógica numérica | unit | unit | ✅ |
| T7 | rollout (inferência) | unit | unit | ✅ |
| T8 | rollout (leakage-crítico) | unit + leakage | unit + leakage | ✅ |
| T9 | figura (visual) | smoke | smoke | ✅ |
| T10 | figura/relatório (P2) | smoke | smoke | ✅ |
| T11 | orquestrador | integration | integration | ✅ |
| T12 | execução real | acceptance | acceptance | ✅ |

Nenhum `Tests: none` por deferimento. ✅

---

## Requirement Traceability

| Requirement / Critério | Tasks |
|---|---|
| EVAL-01 (R0 regen SIR+subset+GNN W12) | T1, T11 |
| EVAL-02 (M1 GNN livre vs SIR semanal) | T5, T7 |
| EVAL-03 (RMSE/música ±IC + Fig.3 + Wilcoxon) | T3, T9, T11 |
| EVAL-04 (recorte hits longos) | T2, T11 |
| EVAL-05 (M2 GNN k vs SIR causal + persist) | T4, T6, T8 |
| EVAL-06 (RMSE + acerto direcional por k/regime) | T3, T11 |
| EVAL-07 (Figs.8/9 — 4 casos) | T9 |
| EVAL-08 (ablation aresta) — P2 | T10 |
| EVAL-09 (importância features) — P2 | T10 |
| EVAL-10 (análogos β/γ/R₀) — P2 | T10 |
| EVAL-11 (artefatos + summary.md) | T11 |
| EVAL-12 (run_phase3 reprodutível) | T11, T12 |
| C1 (run end-to-end) | T11, T12 |
| C2 (mesmo eixo semanal/subset) | T5, T7, T11 |
| C3 (Fig.3 + IC + Wilcoxon) | T3, T9, T11 |
| C4 (M1 success p<0,01) | T7, T5, T3, T12 |
| C5 (M2 RMSE+dir por k) | T6, T8, T3, T11 |
| C6 (M2 ≥2/3 horizontes) | T8, T6, T12 |
| C7 (hits longos) | T2, T11 |
| C8 (Figs.8/9) | T9 |
| C9 (ablation aresta) — P2 | T10 |
| C10 (importância features) — P2 | T10 |
| C11 (β/γ/R₀) — P2 | T10 |
| C12 (summary.md tabela-mãe) | T11, T12 |

**Coverage:** 12 requisitos (EVAL-01–12) e 12 critérios (C1–C12) mapeados a ≥1 task. ✅

---

## Notas de execução

- **Delegação:** Wave 0 (T1–T4) e Wave 2 (T9/T10) são `[P]` → um sub-agente por task. T5→T6 e T7→T8 são sequenciais (mesmo módulo). T11/T12 não delegar em paralelo (precisam do contexto acumulado).
- **Gates antes de gastar tempo:** o smoke do T11 (`--smoke`) é o gate antes da execução cheia do T12 (análogo ao R1.T7→R1.T8 da Phase 2).
- **MCPs/Skills = NONE** em todas as tasks (stack pura, igual Phases 1–2). Se quiser plugar alguma ferramenta, avise antes do Execute.
