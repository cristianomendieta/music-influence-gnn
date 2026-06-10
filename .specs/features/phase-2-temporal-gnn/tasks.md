# Phase 2 — Tasks (Temporal GNN heterogêneo)

**Design:** [`design.md`](design.md) · **Spec:** [`spec.md`](spec.md) · **Context:** [`context.md`](context.md)
**Status:** Draft

> 15 tasks atômicas em 6 waves. Cada task = 1 commit. PR único final referenciando spec/design/tasks.
> Stack PyG puro — **sem MCPs/Skills** necessários (igual à Phase 1).

---

## Execution Plan

### Wave 0 — Deps (sequencial)
```
T1
```

### Wave 1 — Dados/janelas (sequencial; base de todo o resto)
```
T1 → T2 → T3 → T4
```

### Wave 2 — Modelos (paralelo após T1)
```
        ┌→ T5 (encoder)   ─┐
T1 ─────┼→ T6 (head)      ─┼→ T7 (diffusion_gnn)
        └→ T8 (baseline)   ┘   (T7 depende de T5+T6)
```

### Wave 3 — Treino (sequencial após T4+T7)
```
T4,T7 → T9 (train_one) → T10 (run_grid) → T11 (evaluate)
```

### Wave 4 — Testes (paralelo após T7+T4)
```
        ┌→ T12 (leakage C2)
T7,T4 ──┤
        └→ T13 (forward smoke)
```

### Wave 5 — Entrypoint + execução (sequencial)
```
T8,T11 → T14 (run_phase2.py) → T15 (rodar grid + registrar)
```

---

## Parallel Execution Map
```
Wave 0:  T1
Wave 1:  T2 → T3 → T4                 (sequencial)
Wave 2:  T5 [P]  T6 [P]  T8 [P]  → T7 (T7 após T5,T6)
Wave 3:  T9 → T10 → T11               (sequencial)
Wave 4:  T12 [P]  T13 [P]             (após T7,T4)
Wave 5:  T14 → T15                    (sequencial; T15 = execução real)
```

`models/` = `src/music_diffusion_gnn/models/` · `training/` = `src/music_diffusion_gnn/training/` ·
testes = `tests/` (raiz, padrão da Phase 1).

---

## Task Breakdown

### T1: Adicionar `pyarrow` a `pyproject.toml`
**What:** Adicionar `pyarrow>=14.0` em `[project].dependencies` (engine de parquet ausente da `.venv`).
**Where:** `pyproject.toml`
**Depends on:** None
**Reuses:** bloco `dependencies` existente (linhas 15–27)
**Requirement:** R7.3

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pyarrow>=14.0` **adicionado** a `[project].dependencies` (não estava listado — só foi instalado ad-hoc na sessão de design)
- [ ] `.venv/bin/python -c "import pyarrow"` retorna sem erro
- [ ] `.venv/bin/python -c "import pandas as pd; pd.read_parquet('data/processed/timeseries.parquet', columns=['date']).head()"` lê sem erro

**Verify:**
```bash
.venv/bin/python -c "import pyarrow, pandas; print(pyarrow.__version__)"
```

**Commit:** `chore(phase2): pin pyarrow as parquet engine`

> ⚠️ A `.venv` aponta interpreter de path antigo (`music-diffusion-gnn`); usar `.venv/bin/python -m pip`, **não** `.venv/bin/pip` (quebrado). Registrado no STATE.md.

---

### T2: `aggregate_weekly` em `training/dataset.py`
**What:** Função que lê `timeseries.parquet` (diário) e agrega para `(song_id, chart, week, y_week)` por média de `y` na ISO-week; descarta `week>260`.
**Where:** `src/music_diffusion_gnn/training/dataset.py`
**Depends on:** T1
**Reuses:** `week_index()` em [graph/temporal.py:13](../../../src/music_diffusion_gnn/graph/temporal.py#L13)
**Requirement:** R0.1, T-gran (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `aggregate_weekly(ts_df) -> pd.DataFrame` com colunas exatas `[song_id, chart, week, y_week]`
- [ ] `week = week_index(date)`; linhas com `week > 260` removidas
- [ ] `y_week` = média dos `y` diários por `(song_id, chart, week)`
- [ ] `chart` normalizado para `{viral50, top200}` (ou códigos 0/1 documentados)
- [ ] Smoke: agregação preserva ~260 semanas máx por série, `y_week ∈ [0,0.5]`

**Verify:**
```bash
.venv/bin/python -c "
from music_diffusion_gnn.training.dataset import aggregate_weekly
import pandas as pd
df = pd.read_parquet('data/processed/timeseries.parquet')
w = aggregate_weekly(df)
assert {'song_id','chart','week','y_week'} <= set(w.columns)
assert w.week.max() <= 260 and w.y_week.between(0,0.5).all()
print(w.shape, w.week.max())"
```

**Commit:** `feat(phase2): aggregate daily timeseries to weekly targets`

---

### T3: `temporal_split` em `training/dataset.py`
**What:** Divide o DataFrame semanal em train/val/test por data (não por música).
**Where:** `src/music_diffusion_gnn/training/dataset.py` (mesmo módulo, função nova)
**Depends on:** T2
**Reuses:** `week_index()`; constantes de fronteira derivadas das datas do ROADMAP
**Requirement:** R3.1, R3.2

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `temporal_split(weekly_df) -> dict[str, pd.DataFrame]` com chaves `{train, val, test}`
- [ ] train: `week ≤ week_index(2020-06-30)`; val: `week_index(2020-07-01) ≤ week ≤ week_index(2020-12-31)`; test: `week_index(2021-01-01) ≤ week ≤ 260`
- [ ] Fronteiras documentadas como constantes nomeadas no módulo
- [ ] Splits disjuntos por `week`; união = todo o DataFrame (verificado)

**Verify:**
```bash
.venv/bin/python -c "
from music_diffusion_gnn.training.dataset import aggregate_weekly, temporal_split
import pandas as pd
s = temporal_split(aggregate_weekly(pd.read_parquet('data/processed/timeseries.parquet')))
assert set(s)=={'train','val','test'}
assert s['train'].week.max() < s['val'].week.min() < s['val'].week.max() < s['test'].week.min()
print({k:len(v) for k,v in s.items()})"
```

**Commit:** `feat(phase2): temporal train/val/test split by date`

---

### T4: `Sample` + `build_samples` em `training/dataset.py`
**What:** Dataclass `Sample` e função que gera tuplas de treino `(song, chart, target_week)` com janela causal `[w-W..w-1]` left-padded.
**Where:** `src/music_diffusion_gnn/training/dataset.py`
**Depends on:** T3
**Reuses:** `node_id_map.json` (`data/processed/graph/node_id_map.json`); `first_seen_week` do grafo
**Requirement:** R0.2, R0.4, OQ1, OQ3, OQ6 (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `Sample` dataclass com campos do design (`song_idx, chart, target_week, window_weeks[W], pad_mask, y`)
- [ ] `build_samples(weekly_df, W, node_id_map, first_seen) -> list[Sample]`
- [ ] Alvo gerado **só** para `w > first_seen_week` (≥1 semana de história); `w == first_seen_week` não gera amostra
- [ ] `window_weeks` left-padded com `-1` quando `w-k < first_seen_week`; `pad_mask[k]=True` no padding
- [ ] `assert` falha-rápido se `song_id` do subset não tem nó no grafo (C4 Phase 1 garante)
- [ ] `song_idx` mapeado via `node_id_map`; `chart` codificado 0/1

**Verify:**
```bash
.venv/bin/python -c "
from music_diffusion_gnn.training.dataset import aggregate_weekly, temporal_split, build_samples
# (passar node_id_map + first_seen reais); checa janela e padding
print('build_samples importável e tipável')"
```

**Commit:** `feat(phase2): build causal windowed training samples`

---

### T5: `HeteroSpatialEncoder` em `models/encoder.py` [P]
**What:** Encoder espacial HeteroGraphSAGE via `to_hetero(SAGE)` → embedding por nó música.
**Where:** `src/music_diffusion_gnn/models/encoder.py`
**Depends on:** T1
**Reuses:** padrão `to_hetero(SAGE, g.metadata())` do smoke-test da Phase 1 ([scripts/run_phase1.py](../../../scripts/run_phase1.py))
**Requirement:** R1.1

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `HeteroSpatialEncoder.__init__(metadata, hidden, layers, dropout)` constrói `to_hetero(SAGE)`
- [ ] `forward(x_dict, edge_index_dict) -> Tensor` retorna `Z_music ∈ (N_music, hidden)`
- [ ] `layers ∈ {2,3}` e `hidden ∈ {64,128}` configuráveis
- [ ] Forward roda em `mask_until(hetero_full, w)` sem erro de shape

**Verify:**
```bash
.venv/bin/python -c "
import torch
from music_diffusion_gnn.models.encoder import HeteroSpatialEncoder
g = torch.load('data/processed/graph/hetero_full.pt', weights_only=False)
enc = HeteroSpatialEncoder(g.metadata(), hidden=64, layers=2, dropout=0.2)
z = enc(g.x_dict, g.edge_index_dict)
print(z.shape)  # (N_music, 64)"
```

**Commit:** `feat(phase2): HeteroGraphSAGE spatial encoder`

---

### T6: `TemporalHead` em `models/temporal_head.py` [P]
**What:** GRU 1-camada sobre janela de embeddings + MLP → `ŷ ∈ [0,0.5]` com `0.5*sigmoid`.
**Where:** `src/music_diffusion_gnn/models/temporal_head.py`
**Depends on:** T1
**Reuses:** —
**Requirement:** R1.2, R1.3

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `TemporalHead.__init__(hidden, dropout)` = GRU(hidden,hidden,1) + MLP(hidden→hidden→1)
- [ ] `forward(seq: (B,W,hidden), pad_mask: (B,W)) -> (B,)`
- [ ] `pad_mask` respeitado (passos de padding não contaminam o estado final)
- [ ] Saída `= 0.5*sigmoid(.)` → garantidamente em `[0,0.5]`

**Verify:**
```bash
.venv/bin/python -c "
import torch
from music_diffusion_gnn.models.temporal_head import TemporalHead
h = TemporalHead(hidden=64, dropout=0.2)
out = h(torch.randn(8,4,64), torch.zeros(8,4,dtype=torch.bool))
assert out.shape==(8,) and (out>=0).all() and (out<=0.5).all()
print('ok', out.min().item(), out.max().item())"
```

**Commit:** `feat(phase2): GRU+MLP temporal head with [0,0.5] output`

---

### T7: `MusicDiffusionGNN` em `models/diffusion_gnn.py`
**What:** Orquestrador encoder+head; `encode_weeks` (banco por semana via `mask_until`), `predict` (gather de janelas), `count_params`.
**Where:** `src/music_diffusion_gnn/models/diffusion_gnn.py`
**Depends on:** T5, T6
**Reuses:** `mask_until()` em [graph/temporal.py:32](../../../src/music_diffusion_gnn/graph/temporal.py#L32)
**Requirement:** R0.3, R1.4, R2.3, OQ2 (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `encode_weeks(g, weeks) -> dict[int, Tensor]`: para cada semana distinta, `mask_until(g, w)` → encoder, **1× por semana** (cache intra-forward)
- [ ] `predict(bank, samples) -> Tensor`: monta `(B,W,hidden)` por gather em `bank[w][song_idx]`, zera padding, chama a head
- [ ] Snapshots diferenciáveis (fazem parte do autograd do batch)
- [ ] `count_params() -> int`
- [ ] `assert` C4: `count_params() ∈ [50K, 500K]` para config hidden=128,layers=3 (ordem ~200K)

**Verify:**
```bash
.venv/bin/python -c "
import torch
from music_diffusion_gnn.models.diffusion_gnn import MusicDiffusionGNN
g = torch.load('data/processed/graph/hetero_full.pt', weights_only=False)
m = MusicDiffusionGNN(g.metadata(), hidden=128, layers=3, dropout=0.2)
n = m.count_params(); assert 50_000 <= n <= 500_000, n
print('params', n)"
```

**Commit:** `feat(phase2): MusicDiffusionGNN orchestrator with per-week embedding bank`

---

### T8: `persistence_predict` em `models/baselines.py` [P]
**What:** Baseline de persistência ingênua `ŷ(w)=y(w-1)` por split/regime.
**Where:** `src/music_diffusion_gnn/models/baselines.py`
**Depends on:** T1
**Reuses:** —
**Requirement:** R5.1

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `persistence_predict(y_week_df, split) -> np.ndarray` alinhado às mesmas tuplas-alvo do GNN
- [ ] `ŷ(w) = y(w-1)`; primeiro passo do span usa `y` floor (sem leakage do futuro)
- [ ] Funciona por regime (viral50 / top200) separadamente

**Verify:**
```bash
.venv/bin/python -c "
from music_diffusion_gnn.models.baselines import persistence_predict
print('persistence_predict importável')"
```

**Commit:** `feat(phase2): naive persistence baseline`

---

### T9: `train_one` em `training/trainer.py`
**What:** Loop de treino de uma config: Adam(lr, weight_decay), dropout, early stopping no val MSE; retorna best state_dict + curvas + val MSE.
**Where:** `src/music_diffusion_gnn/training/trainer.py`
**Depends on:** T4, T7
**Reuses:** `rmse()` em [evaluation/metrics.py:9](../../../src/music_diffusion_gnn/evaluation/metrics.py#L9); `Config` dataclass (design)
**Requirement:** R4.1, R4.3, R7.2

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `Config` dataclass conforme design (W, hidden, layers, lr, weight_decay, dropout, max_epochs, patience, seed)
- [ ] `train_one(config, splits, g) -> TrainResult` (best_state_dict, train_curve, val_curve, val_mse)
- [ ] Minibatch de tuplas com shuffle por `Generator` semeado (OQ6); `encode_weeks` só nas semanas distintas do batch
- [ ] Early stopping com `patience`; seeds fixas em torch/numpy/random
- [ ] Smoke: 1 config minúscula (W=4,hidden=64,layers=2, max_epochs=2) treina sem erro

**Verify:**
```bash
.venv/bin/python -c "from music_diffusion_gnn.training.trainer import train_one, Config; print('train_one ok')"
```

**Commit:** `feat(phase2): single-config training loop with early stopping`

---

### T10: `run_grid` em `training/trainer.py`
**What:** Itera o grid `W×hidden×layers×lr` (24 configs) chamando `train_one`; tabela de resultados; seleciona melhor por val MSE.
**Where:** `src/music_diffusion_gnn/training/trainer.py`
**Depends on:** T9
**Reuses:** `train_one`
**Requirement:** R4.2, C5

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `run_grid(grid, splits, g) -> pd.DataFrame` com 1 linha por config (config × train_mse × val_mse × params × tempo)
- [ ] Grid = `W∈{4,8,12} × hidden∈{64,128} × layers∈{2,3} × lr∈{1e-3,5e-4}` = **24** linhas
- [ ] Retorna também a melhor config (menor val MSE) e seu best_state_dict
- [ ] DataFrame serializável em parquet (R6.2)

**Verify:**
```bash
.venv/bin/python -c "from music_diffusion_gnn.training.trainer import run_grid; print('run_grid ok')"
```

**Commit:** `feat(phase2): hyperparameter grid runner`

---

### T11: `evaluate` em `training/trainer.py`
**What:** Avaliação nos dois protocolos: forecasting (held-out) e retroativo (reconstrução teacher-forced in-sample); MSE/RMSE por regime; garante `ŷ∈[0,0.5]`.
**Where:** `src/music_diffusion_gnn/training/trainer.py`
**Depends on:** T10
**Reuses:** `rmse()`; `persistence_predict` (T8)
**Requirement:** R2.1, R2.2, C3, C6, C7, C8, OQ4 (design)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `evaluate(model, splits, mode) -> dict` com `mode ∈ {forecasting, retroactive}`
- [ ] forecasting: MSE no val (split futuro held-out) por regime
- [ ] retroactive: reconstrução teacher-forced da curva inteira por música, score em todas as semanas do span
- [ ] `assert (0<=yhat).all() and (yhat<=0.5).all()` (C3)
- [ ] Retorna predições por tupla para serialização (`val_predictions.parquet`, com coluna `mode`)
- [ ] Compara contra `persistence_predict` no mesmo split/regime

**Verify:**
```bash
.venv/bin/python -c "from music_diffusion_gnn.training.trainer import evaluate; print('evaluate ok')"
```

**Commit:** `feat(phase2): dual-protocol evaluation (forecasting + retroactive)`

---

### T12: Teste de leakage `tests/test_phase2_leakage.py` (C2) [P]
**What:** Teste unitário que prova R0.3 — `y(w)` nunca vê aresta com `first_seen_week > w-1`.
**Where:** `tests/test_phase2_leakage.py`
**Depends on:** T7, T4
**Reuses:** `mask_until`, `encode_weeks`, `build_samples`
**Requirement:** C2, R0.3

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] Constrói sample em `w`; verifica que `encode_weeks` só é chamado com semanas `≤ w-1`
- [ ] Verifica que toda aresta com `first_seen_week > w-1` está **ausente** do snapshot `mask_until(g, w-1)`
- [ ] Teste passa: `pytest tests/test_phase2_leakage.py`
- [ ] 0 violação (assertivo, não probabilístico)

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase2_leakage.py -q
```

**Commit:** `test(phase2): assert no temporal leakage in snapshots (C2)`

---

### T13: Smoke test do forward end-to-end [P]
**What:** Teste que monta um minibatch real e roda `encode_weeks`→`predict`, conferindo shapes e range de saída.
**Where:** `tests/test_phase2_forward.py`
**Depends on:** T7, T4
**Reuses:** `MusicDiffusionGNN`, `build_samples`
**Requirement:** C3 (range), sanity de integração

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] Monta ~8 samples reais do split de train; roda forward
- [ ] `ŷ.shape == (B,)`, `ŷ ∈ [0,0.5]`
- [ ] `count_params() ∈ [50K,500K]` para config representativa (C4)
- [ ] `pytest tests/test_phase2_forward.py` passa

**Verify:**
```bash
.venv/bin/python -m pytest tests/test_phase2_forward.py -q
```

**Commit:** `test(phase2): end-to-end forward smoke test`

---

### T14: Entrypoint `scripts/run_phase2.py`
**What:** Orquestra pipeline completo → grid → seleção → avaliação dupla → baseline → artefatos R6.1–R6.5 → checklist C1–C9 + exit 0/1.
**Where:** `scripts/run_phase2.py`
**Depends on:** T8, T11
**Reuses:** todas as funções acima; padrão de `scripts/run_phase1.py`
**Requirement:** R6.1–R6.5, R7.1, C1, C5–C9

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] Carrega `hetero_full.pt` + `timeseries.parquet` → `aggregate_weekly` → `temporal_split` → `build_samples`
- [ ] `run_grid` → seleção da melhor config
- [ ] `evaluate` forecasting + retroativo nos dois regimes; baseline persistência
- [ ] Escreve `results/phase2/`: `best_model.pt` (R6.1), `grid_results.parquet` (R6.2), `val_predictions.parquet` com coluna `mode` (R6.3), `training_curves.png` (R6.4), `summary.md` com tabela GNN vs persistência (R6.5)
- [ ] Imprime checklist C1–C9; `exit 1` se C6 ou C7 falhar (GNN não supera persistência) + instrução de registrar em STATE.md
- [ ] Seed configurável via CLI/constante

**Verify:**
```bash
.venv/bin/python scripts/run_phase2.py  # smoke; conferir criação de results/phase2/*
```

**Commit:** `feat(phase2): run_phase2 entrypoint with R6 artifacts and C1-C9 checklist`

---

### T15: Rodar grid completo + registrar resultados
**What:** Execução real end-to-end; conferir C1–C9 (especialmente C6/C7); registrar resultados/desvios em STATE.md.
**Where:** execução + `.specs/project/STATE.md`
**Depends on:** T14
**Reuses:** —
**Requirement:** C1–C9, acceptance test

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `python scripts/run_phase2.py` roda end-to-end sem erro (C1); tempo medido (< algumas horas)
- [ ] C2–C9 verdes; `summary.md` mostra GNN < persistência no val nos dois regimes
- [ ] Artefatos R6.1–R6.5 presentes em `results/phase2/`
- [ ] STATE.md atualizado: melhor config, val MSE GNN vs persistência, params, tempo, desvios
- [ ] Se C6/C7 falharem: registrar em STATE.md e acionar decisão sobre Plano B (fora do escopo — só registrar)

**Verify:**
```bash
.venv/bin/python scripts/run_phase2.py && ls results/phase2/
```

**Commit:** `chore(phase2): record grid results and phase outcome in STATE`

---

## Task Granularity Check

| Task | Escopo | Status |
|------|--------|--------|
| T1 confirmar pyarrow | 1 dep | ✅ |
| T2 aggregate_weekly | 1 função | ✅ |
| T3 temporal_split | 1 função | ✅ |
| T4 Sample+build_samples | 1 dataclass + 1 função coesa | ✅ |
| T5 encoder | 1 classe | ✅ |
| T6 temporal_head | 1 classe | ✅ |
| T7 diffusion_gnn | 1 classe (3 métodos coesos) | ✅ |
| T8 baseline | 1 função | ✅ |
| T9 train_one | 1 função + Config | ✅ |
| T10 run_grid | 1 função | ✅ |
| T11 evaluate | 1 função (2 modos) | ✅ |
| T12 leakage test | 1 arquivo de teste | ✅ |
| T13 forward smoke | 1 arquivo de teste | ✅ |
| T14 run_phase2 | 1 entrypoint | ✅ |
| T15 execução | run + registro | ✅ |

---

## Requirement Traceability

| Requirement | Tasks |
|---|---|
| R0.1 alvo y_week | T2 |
| R0.2 janela de embeddings | T4, T7 |
| R0.3 sem leakage | T7, T12 |
| R0.4 universo subset | T4 |
| R1.1 HeteroSAGE | T5 |
| R1.2 GRU | T6 |
| R1.3 head [0,0.5] | T6 |
| R1.4 ~200K params | T7, T13 |
| R2.1/R2.2/R2.3 dois objetivos | T11 |
| R3.1/R3.2/R3.3 splits | T3, T4 |
| R4.1 Adam/dropout/early-stop | T9 |
| R4.2 grid | T10 |
| R4.3 seeds/métricas | T9, T10 |
| R5.1 persistência | T8 |
| R5.2 critério central | T11, T15 |
| R6.1–R6.5 artefatos | T14 |
| R7.1–R7.3 reprodutibilidade | T1, T9, T14 |
| C1 | T15 |
| C2 | T12 |
| C3 | T6, T11, T13 |
| C4 | T7, T13 |
| C5 | T10 |
| C6/C7 | T11, T15 |
| C8 | T11, T14 |
| C9 | T14 |
