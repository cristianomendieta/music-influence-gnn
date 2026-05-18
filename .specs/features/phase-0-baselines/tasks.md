# Phase 0 — Tasks

> Tasks atômicas, sequenciais, com verificação. Cada uma vira um commit.
> Spec: [`spec.md`](spec.md). Design: [`design.md`](design.md).
> **Fase concluída em 2026-05-12.** Wave C (wave-based) descartada por decisão do pesquisador — ver STATE.md (2026-05-12).

## Convenções

- **Status:** `[ ]` pending · `[~]` skipped/descartado · `[x]` done · `[!]` blocked

## Wave A — Setup e dados (paralelizáveis aos pares)

### T0.1 — Instalar dependências e validar ambiente

- **Status:** `[x]`
- **Cobre:** R5.3
- **Arquivos:** —
- **Ações:**
  - `pip install -e .[dev]`
  - `python -c "import torch, torch_geometric, scipy, networkx; print('ok')"`
  - `python --version` registrado em STATE.md.
- **Done quando:** import sem erro; versão de Python anotada.

### T0.2 — Loaders de dados crus

- **Status:** `[x]`
- **Cobre:** R0.2 (parcial, só leitura)
- **Arquivos:** `src/music_diffusion_gnn/data/loaders.py`
- **Ações:** funções `load_charts()`, `load_songs()`, `load_artists()`, `load_genre_network()`.
  Tipos garantidos (datas como `datetime`, ids como `str`).
- **Done quando:**
  - Imports funcionam de fora do pacote (`from music_diffusion_gnn.data.loaders import load_charts`).
  - `load_charts()` retorna DataFrame com `song_id` extraído da URL e `chart ∈ {top200, viral50}`.

### T0.3 — Construir subset viral∩hit e persistir

- **Status:** `[x]`
- **Cobre:** R1.1, R1.2, R1.3
- **Arquivos:** `src/music_diffusion_gnn/data/subset.py`,
  `data/processed/subset_ids.json` (gerado).
- **Resultado:** 1.981 músicas (paper: 1.977; diferença de 4 músicas por diferença de período → declarada como limitação).

### T0.4 — Pré-processamento e parquet de séries temporais

- **Status:** `[x]`
- **Cobre:** R0.1, R0.2, R0.3
- **Arquivos:** `src/music_diffusion_gnn/data/preprocess.py`,
  `data/processed/timeseries.parquet` (gerado).
- **Resultado:** parquet com séries rank score → MA-7d → min-max [0, 0.5] → floor 0.001.

### T0.5 — Teste unitário de preprocess

- **Status:** `[x]`
- **Cobre:** R0.1
- **Arquivos:** `tests/test_preprocess.py`
- **Resultado:** 6 casos (MA-7d, min-max, floor, edge cases) — todos verdes.

## Wave B — Baseline SIR

### T0.6 — Implementar SIR fit

- **Status:** `[x]`
- **Cobre:** R2.1, R2.2, R2.3, R2.4
- **Arquivos:** `src/music_diffusion_gnn/baselines/sir.py`
- **Resultado:** `SIRFit` dataclass + `fit_sir()` com `odeint` + `curve_fit`.

### T0.7 — Teste sintético do SIR

- **Status:** `[x]`
- **Cobre:** R2 (validação)
- **Arquivos:** `tests/test_sir.py`
- **Resultado:** 6 casos — recuperação de β/γ dentro de 5% em dados sintéticos.

### T0.8 — Paralelizador genérico

- **Status:** `[x]`
- **Cobre:** D6
- **Arquivos:** `src/music_diffusion_gnn/baselines/parallel.py`
- **Resultado:** `fit_all()` com `joblib.Parallel`, retorna DataFrame indexado por `(song_id, chart)`.

### T0.9 — Rodar SIR no subset completo

- **Status:** `[x]`
- **Cobre:** R2 (execução)
- **Arquivos:** `results/phase0/sir_params.parquet` (gerado)
- **Resultado:** 100% de convergência; parâmetros β e γ ajustados para 1.981 × 2 charts.

## Wave C — Baseline Wave-based *(DESCARTADO)*

### T0.10 — Implementar wave-based fit

- **Status:** `[~]`
- **Motivo:** Descartado por decisão do pesquisador em 2026-05-12. `differential_evolution` com M_max=5 levou >15h; M_max=3 não justifica implementação adicional dado que o SIR já passou todos os critérios de aceitação R4. Decisão registrada em STATE.md.

### T0.11 — Teste sintético do wave-based

- **Status:** `[~]`
- **Motivo:** Descartado junto com T0.10.

### T0.12 — Rodar wave-based no subset completo

- **Status:** `[~]`
- **Motivo:** Descartado junto com T0.10.

## Wave D — Avaliação e fechamento

### T0.13 — Métricas (RMSE médio, IC95%, Mann-Whitney)

- **Status:** `[x]`
- **Cobre:** R4
- **Arquivos:** `src/music_diffusion_gnn/evaluation/metrics.py`, `tests/test_metrics.py`
- **Resultado:** 6 casos — RMSE e Mann-Whitney U testados e verdes.

### T0.14 — Relatório `summary.md` + boxplot

- **Status:** `[x]`
- **Cobre:** R4, R6
- **Arquivos:** `src/music_diffusion_gnn/evaluation/report.py`,
  `results/phase0/summary.md`, `results/phase0/boxplot_fig3.png`
- **Resultado:** tabela R4 preenchida; boxplot gerado (5/5 critérios ✅).

### T0.15 — Script orquestrador `run_phase0.py`

- **Status:** `[x]`
- **Cobre:** R5.1
- **Arquivos:** `scripts/run_phase0.py`
- **Resultado:** execução idempotente; segunda chamada usa cache de `data/processed/` e `results/phase0/`.

### T0.16 — Verificar critérios numéricos da R4

- **Status:** `[x]`
- **Cobre:** R4
- **Resultados obtidos (2026-05-12):**
  - SIR RMSE virality: **0.0289** ✅
  - SIR RMSE success: **0.0471** ✅
  - Mann-Whitney p-value: **1.61e-39** ✅
  - Subset: **1.981 músicas** ✅
  - Convergência SIR: **100%** ✅

### T0.17 — Atualizar STATE.md e fechar fase

- **Status:** `[x]`
- **Cobre:** —
- **Resultado:** STATE.md atualizado; ROADMAP.md marcado como completed; fase encerrada em 2026-05-17.

## Dependências entre tasks

```
T0.1 ─┬─> T0.2 ─> T0.3 ─> T0.4 ─> T0.5
      │                      │
      │                      ├─> T0.6 ─> T0.7 ─> T0.8 ─> T0.9 ──┐
      │                      │                                  │
      │                      └─> [T0.10-T0.12 DESCARTADOS]        │
      │                                                        ▼
      └──────────────────────────────────────────> T0.13 ─> T0.14 ─> T0.15 ─> T0.16 ─> T0.17
```

## Critério global de done da fase

✅ **FASE CONCLUÍDA** — T0.16 confirmou os 5 critérios numéricos da R4 (ver `results/phase0/summary.md`).
