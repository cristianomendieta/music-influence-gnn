# Phase 0 — Tasks

> Tasks atômicas, sequenciais, com verificação. Cada uma vira um commit.
> Spec: [`spec.md`](spec.md). Design: [`design.md`](design.md).

## Convenções

- **Status:** `[ ]` pending · `[~]` in_progress · `[x]` done · `[!]` blocked
- **Cada task** lista: arquivos tocados, requisito coberto (R-id), critério de done.
- **PR único no fim:** todas as tasks compõem um único PR atômico ao final da fase.

## Wave A — Setup e dados (paralelizáveis aos pares)

### T0.1 — Instalar dependências e validar ambiente

- **Status:** `[ ]`
- **Cobre:** R5.3
- **Arquivos:** —
- **Ações:**
  - `pip install -e .[dev]`
  - `python -c "import torch, torch_geometric, scipy, networkx; print('ok')"`
  - `python --version` registrado em STATE.md.
- **Done quando:** import sem erro; versão de Python anotada.

### T0.2 — Loaders de dados crus

- **Status:** `[ ]`
- **Cobre:** R0.2 (parcial, só leitura)
- **Arquivos:** `src/music_diffusion_gnn/data/loaders.py`
- **Ações:** funções `load_charts()`, `load_songs()`, `load_artists()`, `load_genre_network()`.
  Tipos garantidos (datas como `datetime`, ids como `str`).
- **Done quando:**
  - Imports funcionam de fora do pacote (`from music_diffusion_gnn.data.loaders import load_charts`).
  - `load_charts()` retorna DataFrame com `song_id` extraído da URL e `chart ∈ {top200, viral50}`.

### T0.3 — Construir subset viral∩hit e persistir

- **Status:** `[ ]`
- **Cobre:** R1.1, R1.2, R1.3
- **Arquivos:** `src/music_diffusion_gnn/data/subset.py`,
  `data/processed/subset_ids.json` (gerado).
- **Ações:**
  - `build_subset()` cruza `top200_ids ∩ viral50_ids` e filtra por presença
    de features acústicas em `songs`.
  - Persiste `data/processed/subset_ids.json`.
- **Done quando:**
  - JSON existe com `n == 1179`.
  - Todas as 1.179 ids têm features acústicas (asserção no código).

### T0.4 — Pré-processamento e parquet de séries temporais

- **Status:** `[ ]`
- **Cobre:** R0.1, R0.2, R0.3
- **Arquivos:** `src/music_diffusion_gnn/data/preprocess.py`,
  `data/processed/timeseries.parquet` (gerado).
- **Ações:**
  - `build_timeseries()`: rank score → MA-7d → min-max [0, 0.5] → floor 0.001.
  - Long format: `song_id | chart | date | rank_score | y`.
  - Restringir ao subset.
- **Done quando:**
  - Parquet existe com ~1179 × 2 × 1826 ≈ 4,3M linhas.
  - Para uma música conhecida, valores conferem com cálculo manual em notebook.

### T0.5 — Teste unitário de preprocess

- **Status:** `[ ]`
- **Cobre:** R0.1
- **Arquivos:** `tests/test_preprocess.py`
- **Ações:** série sintética de entrada → checar MA-7d, min-max e floor.
- **Done quando:** `pytest tests/test_preprocess.py` verde.

## Wave B — Baseline SIR

### T0.6 — Implementar SIR fit

- **Status:** `[ ]`
- **Cobre:** R2.1, R2.2, R2.3, R2.4
- **Arquivos:** `src/music_diffusion_gnn/baselines/sir.py`
- **Ações:**
  - `SIRFit` dataclass conforme design D3.
  - `fit_sir()` com `odeint` + `curve_fit`, initial guess (0.5, 0.5),
    bounds `[(0, 0), (10, 10)]`.
  - Calcular RMSE no domínio normalizado.
- **Done quando:** assinatura confere com design.md §"Contratos".

### T0.7 — Teste sintético do SIR

- **Status:** `[ ]`
- **Cobre:** R2 (validação)
- **Arquivos:** `tests/test_sir.py`
- **Ações:** gerar curva SIR com (β=0.3, γ=0.1) conhecidos → rodar `fit_sir`
  → checar recuperação dentro de 5%.
- **Done quando:** teste verde.

### T0.8 — Paralelizador genérico

- **Status:** `[ ]`
- **Cobre:** D6
- **Arquivos:** `src/music_diffusion_gnn/baselines/parallel.py`
- **Ações:** `fit_all(timeseries, fit_fn, n_jobs=-1)` com `joblib.Parallel`.
  Retorna DataFrame indexado por `(song_id, chart)`.
- **Done quando:** smoke-test com 10 músicas roda sem erro em <30s.

### T0.9 — Rodar SIR no subset completo

- **Status:** `[ ]`
- **Cobre:** R2 (execução)
- **Arquivos:** `results/phase0/sir_params.parquet` (gerado),
  `results/phase0/rmse_per_song.parquet` (parcial: SIR).
- **Ações:** `fit_all(timeseries_subset, fit_sir)` → persistir parquet.
- **Done quando:**
  - Parquet com ~2.358 linhas (1.179 × 2 charts).
  - Coluna `converged == True` em ≥95% das linhas.

## Wave C — Baseline Wave-based

### T0.10 — Implementar wave-based fit

- **Status:** `[ ]`
- **Cobre:** R3.1, R3.2, R3.3
- **Arquivos:** `src/music_diffusion_gnn/baselines/wave_based.py`
- **Ações:**
  - `WaveFit` dataclass conforme design D4.
  - Para cada M ∈ {1, ..., 5}: `differential_evolution` (`maxiter=200, popsize=12`)
    + refino via `curve_fit`. Selecionar melhor M por **BIC**.
- **Done quando:** smoke-test com 1 música roda em <60s.

### T0.11 — Teste sintético do wave-based

- **Status:** `[ ]`
- **Cobre:** R3 (validação)
- **Arquivos:** `tests/test_wave_based.py`
- **Ações:** soma de 2 SIRs com offsets distintos → fit recupera M=2.
- **Done quando:** teste verde.

### T0.12 — Rodar wave-based no subset completo

- **Status:** `[ ]`
- **Cobre:** R3 (execução)
- **Arquivos:** `results/phase0/wave_params.parquet`,
  `results/phase0/rmse_per_song.parquet` (atualizado: + wave).
- **Ações:** `fit_all(timeseries_subset, fit_wave)`.
- **Done quando:**
  - Parquet com ~2.358 linhas.
  - Tempo total ≤ 3h em laptop (se exceder, reduzir M_max para 4 e registrar em STATE.md).

## Wave D — Avaliação e fechamento

### T0.13 — Métricas (RMSE médio, IC95%, Mann-Whitney)

- **Status:** `[ ]`
- **Cobre:** R4
- **Arquivos:** `src/music_diffusion_gnn/evaluation/metrics.py`,
  `tests/test_metrics.py`
- **Done quando:** Mann-Whitney pareado com `scipy.stats.mannwhitneyu`;
  teste contra valores conhecidos verde.

### T0.14 — Relatório `summary.md` + boxplot

- **Status:** `[ ]`
- **Cobre:** R4, R6
- **Arquivos:** `src/music_diffusion_gnn/evaluation/report.py`,
  `results/phase0/summary.md`, `results/phase0/boxplot_fig3.png`
- **Done quando:**
  - Tabela R4 preenchida no markdown.
  - Boxplot replica visualmente Fig. 3 do paper (mesmo layout, mesmas categorias).

### T0.15 — Script orquestrador `run_phase0.py`

- **Status:** `[ ]`
- **Cobre:** R5.1
- **Arquivos:** `scripts/run_phase0.py`
- **Ações:** sequência idempotente: subset → timeseries → SIR → wave → métricas → report.
  Usa `if not exists` para reaproveitar artefatos já gerados.
- **Done quando:** rodar duas vezes seguidas; segunda execução em <30s
  (tudo cacheado).

### T0.16 — Verificar critérios numéricos da R4

- **Status:** `[ ]`
- **Cobre:** R4
- **Arquivos:** `results/phase0/summary.md` (atualizado)
- **Ações:** confirmar que SIR RMSE virality ∈ [0.025, 0.031] e success ∈ [0.047, 0.057];
  Mann-Whitney p-value ordem 1e-60; wave-based ≤ SIR.
- **Done quando:**
  - Os três alvos batidos. Se algum não bater, **bloquear avanço** e registrar em STATE.md.

### T0.17 — Atualizar STATE.md e fechar fase

- **Status:** `[ ]`
- **Cobre:** —
- **Arquivos:** `.specs/project/STATE.md`, `.specs/features/phase-0-baselines/spec.md`
- **Ações:** mover decisões finais para STATE; mudar status da Phase 0 em
  ROADMAP.md para `done`; abrir spec da Phase 1.
- **Done quando:** ROADMAP.md mostra Phase 0 done; Phase 1 spec criada.

## Dependências entre tasks

```
T0.1 ─┬─> T0.2 ─> T0.3 ─> T0.4 ─> T0.5
      │                      │
      │                      ├─> T0.6 ─> T0.7 ─> T0.8 ─> T0.9 ──┐
      │                      │                                  │
      │                      └─> T0.10 ─> T0.11 ─> T0.12 ──────┐│
      │                                                        ▼▼
      └──────────────────────────────────────────> T0.13 ─> T0.14 ─> T0.15 ─> T0.16 ─> T0.17
```

T0.6/T0.7 e T0.10/T0.11 são paralelizáveis (não dependem entre si).
T0.9 e T0.12 também (modelos independentes).

## Critério global de done da fase

Todos os 17 itens em `[x]` **e** T0.16 confirmando os 3 alvos numéricos da R4.
