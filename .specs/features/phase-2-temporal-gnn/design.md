# Phase 2 — Design (Temporal GNN heterogêneo)

**Spec:** [`spec.md`](spec.md) · **Context:** [`context.md`](context.md)
**Status:** Revisão R1 (2026-06-23) — injeção de popularidade defasada
**Depende de:** Phase 1 (`hetero_full.pt`, `mask_until`, `week_index`) + Phase 0 (`timeseries.parquet`)
**Bloqueia:** Phase 3 (avaliação dupla)

> **⚠️ Revisão R1 (2026-06-23):** a 1ª implementação (v1) **perdeu para a persistência
> ingênua em todas as 24 configs do grid** (C6/C7 reprovados). Causa-raiz e correção
> aprovada estão na seção **[Revisão R1](#revisão-r1-2026-06-23--injeção-de-popularidade-defasada)**
> no fim deste arquivo. As seções abaixo descrevem a arquitetura v1 (mantidas como
> registro); leia a R1 para o que muda.

---

## Fato decisivo: granularidade semanal (forçada pelos dados)

`timeseries.parquet` é **diário** (4.44M linhas, 1.981 músicas × 2 charts, `y∈[0,0.5]`,
2017-01-01 → 2022-03-13). O grafo é **semanal**: `first_seen_week ∈ [0,260]` e
`mask_until` opera por semana. O grafo **não consegue** produzir snapshots distintos
por dia (a resolução temporal das arestas é semanal). Logo:

- **Alvo modelado em granularidade semanal.** `y_week(song,chart,w)` = média dos `y`
  diários dentro da ISO-week `w` (`week_index(date)`). ~260 passos por série em vez de ~1.900.
- Linhas de 2022 (`week_index > 260`) são **descartadas** (fora do range do grafo;
  coincide com a limitação dos ~2,5 meses já declarada).
- ⚠️ **Comparabilidade com SIR (Phase 3):** o SIR foi ajustado na série diária. A Phase 3
  fará a ponte (upsample da predição semanal para diária, ou recomputo do RMSE do SIR
  em base semanal). Registrado como decisão T-gran abaixo; não bloqueia a Phase 2.

---

## Resolução das open questions

| # | Questão | Decisão | Justificativa |
|---|---------|---------|---------------|
| OQ1 | Semanas fora do chart (`y=floor`) | **Treinar no span ativo** `[first_seen_week .. last_seen_week]` de cada (song,chart), incluindo semanas de baixa popularidade | A queda/saída do chart é sinal de difusão (o SIR modela a curva inteira). Semanas antes de `first_seen` não têm nó com arestas → excluídas. |
| OQ2 | Custo do encoder por ~260 semanas | **Banco de embeddings por semana, computado por minibatch.** Para um batch de alvos, computar `HeteroSAGE` só nas semanas distintas exigidas pelas janelas; cachear por semana dentro do forward; backprop através desses snapshots | Evita 260 forwards/época indiscriminados. Distintas por batch ≈ B+W. Memória controlada pelo tamanho do batch. |
| OQ3 | Janela no início (`t < W`) | **Left-pad com zeros + máscara**; alvo só para `w > first_seen_week` (≥1 semana de história) | Antes de `first_seen` o nó não tem arestas; zero-pad + máscara no GRU evita viés. |
| OQ4 | Definição do fit retroativo | **Um único modelo, dois protocolos de avaliação.** Forecasting = MSE em split futuro held-out (val/test). Retroativo = reconstrução teacher-forced da curva inteira por música (janelas de valores reais, score em todas as semanas do span) | Mantém R2.3 (encoder compartilhado). Retroativo ≈ ajuste in-sample do SIR; comparação fina na Phase 3. |
| OQ5 | Edge subsampling (anti-overfit) | **Deferir.** v1 usa grafo completo nos snapshots (~700K arestas, OK em CPU); regularização via dropout + weight decay + early stopping (R4.1) | Mantém o pipeline simples. Subsampling de `cotrajectory` (664K arestas) é a alavanca do Plano B se houver overfitting. |
| OQ6 | Batching causal | **Minibatch = amostra de tuplas `(song, chart, target_week)`** do split de treino; shuffle livre | Cada tupla carrega sua própria janela causal (`mask_until(w-1)`); embaralhar tuplas não vaza futuro. Agrupar por semanas necessárias dá eficiência. |

---

## Architecture Overview

```mermaid
graph TD
    G[hetero_full.pt] --> ENC[HeteroSpatialEncoder<br/>HeteroSAGE via to_hetero]
    TS[timeseries.parquet<br/>diário] --> AGG[aggregate_weekly<br/>y_week por song,chart,w]
    AGG --> SPL[temporal_split<br/>train/val/test por data]
    SPL --> SAMP[WindowSampler<br/>tuplas (song,chart,w) + janela W]

    SAMP -->|semanas distintas do batch| BANK[encode_weeks<br/>mask_until(w) → HeteroSAGE<br/>banco {w: Z_music}]
    G --> BANK
    BANK -->|gather song,w-W..w-1| SEQ[sequência de embeddings]
    SEQ --> GRU[GRU 1 camada]
    GRU --> HEAD[MLP → 0.5·sigmoid]
    HEAD --> YHAT[ŷ_week ∈ 0,0.5]

    YHAT --> LOSS[MSE vs y_week]
    YHAT --> EVAL[forecasting + retroativo]
    BASE[persistence ŷ=y_w-1] --> EVAL
    EVAL --> ART[results/phase2/*]
```

**Princípios:**
- **Encoder espacial e cabeça temporal desacoplados.** O encoder produz embedding de música
  por semana; a cabeça (GRU+MLP) consome a janela. Permite cachear snapshots por semana.
- **Snapshot = `mask_until(g, w)` → `HeteroSAGE` forward** (todos os nós música de uma vez).
  Reaproveitado para todas as tuplas do batch que precisam da semana `w`.
- **Sem leakage:** alvo `y(w)` só usa snapshots de semanas `≤ w-1`. Teste unitário (C2).
- **Determinismo:** seeds fixas em torch/numpy/random; amostragem de batch com `Generator` semente.

---

## Code Reuse Analysis

| Componente | Localização | Uso |
|------------|-------------|-----|
| `mask_until(g, w)` | [graph/temporal.py:32](../../../src/music_diffusion_gnn/graph/temporal.py#L32) | Snapshot semanal sem leakage |
| `week_index(date)` | [graph/temporal.py:13](../../../src/music_diffusion_gnn/graph/temporal.py#L13) | Mapear `date → w ∈ [0,260]`; descartar `>260` |
| `hetero_full.pt` | `data/processed/graph/` | Grafo base; `g.metadata()` para `to_hetero` |
| `rmse()` | [evaluation/metrics.py:9](../../../src/music_diffusion_gnn/evaluation/metrics.py#L9) | Métrica val/test |
| smoke-test HeteroSAGE | [scripts/run_phase1.py](../../../scripts/run_phase1.py) | Padrão `to_hetero(SAGE, g.metadata())` já validado (C8 da Phase 1) |
| `load_subset()` | [data/subset.py](../../../src/music_diffusion_gnn/data/subset.py) | Universo das 1.981 músicas-alvo |

**Integração:** `models/` e `training/` estão **vazios** (só `__init__.py`) — Phase 2 os preenche.
`pyproject.toml` precisa de **`pyarrow`** (a `.venv` não tem engine de parquet; já instalado nesta sessão, fixar como dep).

---

## Components

### `models/encoder.py` — `HeteroSpatialEncoder`

- **Purpose:** Embedding por nó música a partir de um snapshot do grafo.
- **Interfaces:**
  - `__init__(metadata, hidden: int, layers: int, dropout: float)` — constrói `to_hetero(SAGE)`.
  - `forward(x_dict, edge_index_dict) -> Tensor` — retorna `Z_music ∈ (N_music, hidden)`.
- **Dependencies:** `torch_geometric.nn.SAGEConv`, `to_hetero`.
- **Reuses:** mesmo padrão do smoke-test da Phase 1.

### `models/temporal_head.py` — `TemporalHead`

- **Purpose:** GRU sobre janela de embeddings + MLP → `ŷ ∈ [0,0.5]`.
- **Interfaces:**
  - `__init__(hidden: int, dropout: float)` — GRU(hidden, hidden, 1) + MLP(hidden→hidden→1).
  - `forward(seq: Tensor(B, W, hidden), pad_mask: Tensor(B, W)) -> Tensor(B,)` — saída `0.5*sigmoid(.)`.
- **Dependencies:** `torch.nn.GRU`.

### `models/diffusion_gnn.py` — `MusicDiffusionGNN`

- **Purpose:** Orquestra encoder + cabeça; expõe `encode_weeks` cacheável.
- **Interfaces:**
  - `encode_weeks(g, weeks: list[int]) -> dict[int, Tensor]` — `{w: Z_music}` via `mask_until` (computa cada semana 1×).
  - `predict(bank, samples) -> Tensor` — monta sequências `(B,W,hidden)` por gather + chama a cabeça.
  - `count_params() -> int`.
- **Dependencies:** encoder, temporal_head.

### `models/baselines.py` — persistência

- **Interfaces:** `persistence_predict(y_week_df, split) -> np.ndarray` — `ŷ(w)=y(w-1)` (primeiro passo do span = `y` floor).

### `training/dataset.py` — janelas e alvos

- **Interfaces:**
  - `aggregate_weekly(ts_df) -> pd.DataFrame` — `(song_id, chart, week, y_week)`; média dos `y` diários por ISO-week; descarta `week>260`.
  - `temporal_split(weekly_df) -> dict[str, DataFrame]` — train `w≤week_index(2020-06-30)`, val `…≤2020-12`, test `2021`.
  - `build_samples(weekly_df, W) -> list[Sample]` — `Sample(song_idx, chart, target_week, window_weeks[W], pad_mask, y)`; alvo só para `w>first_seen_week`.
- **Dependencies:** pandas, `week_index`, `node_id_map.json` (song_id→idx).

### `training/trainer.py` — loop + grid

- **Interfaces:**
  - `train_one(config, splits, g) -> TrainResult` — Adam(lr, weight_decay), early stopping no val MSE, dropout; retorna best state_dict + curvas + val MSE.
  - `run_grid(grid, splits, g) -> pd.DataFrame` — itera `W×hidden×layers×lr`, retorna tabela; seleciona melhor por val MSE.
  - `evaluate(model, splits, mode) -> dict` — `mode ∈ {forecasting, retroactive}`; MSE/RMSE por regime.
- **Dependencies:** models, dataset, `rmse()`.

### `scripts/run_phase2.py` — entrypoint

- **Comportamento:** carrega grafo+série → `aggregate_weekly` → `temporal_split` → `run_grid`
  → seleciona melhor → avalia forecasting+retroativo → baseline persistência →
  escreve R6.1–R6.5 → imprime checklist C1–C9 + exit 0/1.

---

## Data Models

### Sample (uma unidade de treino)
```python
@dataclass
class Sample:
    song_idx: int          # índice PyG do nó música
    chart: int             # 0=viral50, 1=top200
    target_week: int       # w ∈ [1,260]
    window_weeks: list[int]  # [w-W .. w-1], left-padded com -1
    pad_mask: list[bool]   # True onde é padding
    y: float               # y_week observado ∈ [0,0.5]
```

### Config do grid
```python
@dataclass
class Config:
    W: int          # {4,8,12}
    hidden: int     # {64,128}
    layers: int     # {2,3}
    lr: float       # {1e-3,5e-4}
    weight_decay: float = 1e-5
    dropout: float = 0.2
    max_epochs: int = 100
    patience: int = 10
    seed: int = 42
```

### Banco de embeddings (intra-forward)
`dict[int, Tensor(N_music, hidden)]` — uma entrada por semana distinta do minibatch; descartado após o backward.

---

## Error Handling Strategy

| Cenário | Tratamento | Impacto |
|---------|------------|---------|
| `date` de 2022 (`week_index>260`) | Descartar linha em `aggregate_weekly` | Fora do range do grafo; limitação já declarada |
| Música do subset sem nó no grafo | Não deveria ocorrer (C4 da Phase 1 garante); `assert` em `build_samples` | Falha fast |
| Janela vazia (`w == first_seen_week`) | Sample não gerado (precisa ≥1 semana de história) | Sem leakage, sem amostra inútil |
| Semana de janela `< 0` | Left-pad com `-1` + `pad_mask=True`; GRU ignora via mask | Início da série tratado |
| Saída fora de `[0,0.5]` | Garantida por `0.5*sigmoid`; `assert` no `evaluate` (C3) | 0 violação |
| OOM no banco de embeddings | Reduzir batch size; banco é por-batch, não global | Controlável |
| GNN não supera persistência (C6/C7) | `run_phase2.py` exit 1 + registrar em STATE.md; acionar Plano B (HGT/Transformer) | Bloqueia conclusão |
| `pyarrow` ausente | Fixado em `pyproject.toml`; erro claro se faltar | Build falha cedo |

---

## Tech Decisions (não-óbvias)

| Decisão | Escolha | Racional |
|---------|---------|----------|
| T-gran | Alvo **semanal** (`y_week` = média diária por ISO-week) | Grafo é semanal; encoder tratável (~260 vs ~1900 passos). Ponte com SIR diário fica na Phase 3 |
| Cache de snapshots | Banco `{w: Z_music}` por minibatch, não global nem entre épocas | Pesos mudam a cada passo → não dá pra cachear entre épocas; dentro do batch evita recomputo de semanas compartilhadas |
| Um modelo, dois protocolos | Forecasting (held-out) e retroativo (reconstrução in-sample) compartilham encoder+pesos | Cumpre D1/R2.3 sem treinar dois modelos |
| Edge subsampling | Deferido (grafo completo no v1) | 700K arestas OK em CPU; dropout/wd/early-stop bastam de início |
| Saída `[0,0.5]` | `0.5*sigmoid` | Garante range do alvo sem clamp pós-hoc |
| Split por data | train≤2020-06, val 2020-07..12, test 2021 | ROADMAP; test reservado p/ Phase 3 |
| Banco de embeddings diferenciável | Snapshots fazem parte do grafo de autograd do batch | Permite treinar o encoder end-to-end |
| Regime de janela | Left-pad zeros + `pad_mask` | GRU lida com início de série sem viés |

---

## Validação dos critérios (C1–C9)

| Critério | Onde valida | Como |
|----------|-------------|------|
| C1 (end-to-end CPU) | `run_phase2.py` | roda grid completo sem erro, mede tempo |
| C2 (sem leakage) | `tests/test_phase2_leakage.py` | constrói sample em `w`, verifica que `encode_weeks` só recebe semanas `≤ w-1`; aresta com `first_seen_week>w-1` ausente do snapshot |
| C3 (`ŷ∈[0,0.5]`) | `trainer.evaluate` | `assert (0<=yhat).all() and (yhat<=0.5).all()` |
| C4 (params ∈[50K,500K]) | `model.count_params()` | assert na melhor config |
| C5 (grid completo) | `run_grid` | 3×2×2×2 = 24 configs em `grid_results.parquet` |
| C6 (GNN>persist. virality) | `evaluate` | `val_mse_gnn < val_mse_persist` no viral50 |
| C7 (GNN>persist. success) | `evaluate` | idem no top200 |
| C8 (ambos objetivos no val) | `run_phase2.py` | `val_predictions.parquet` tem colunas `mode∈{forecasting,retroactive}` |
| C9 (`summary.md`) | `run_phase2.py` | arquivo com tabela GNN vs persistência |

---

## Plano de execução (preview p/ tasks.md)

1. Deps: `pyarrow` no `pyproject.toml`.
2. `training/dataset.py`: `aggregate_weekly`, `temporal_split`, `build_samples` (+ testes).
3. `models/encoder.py`, `temporal_head.py`, `diffusion_gnn.py`, `baselines.py`.
4. `training/trainer.py`: `train_one`, `run_grid`, `evaluate`.
5. `tests/test_phase2_leakage.py` (C2) + smoke do forward.
6. `scripts/run_phase2.py`: grid → seleção → avaliação → artefatos R6 → checklist C1–C9.
7. Rodar grid, conferir C6/C7; registrar resultados em STATE.md.

---

## Open issues / dívida técnica

1. Comparabilidade semanal↔diária com SIR resolvida só na Phase 3 (T-gran).
2. Custo de memória do banco de snapshots depende de B×(B+W) semanas distintas — medir na 1ª execução; reduzir batch se OOM.
3. Se nenhuma config superar persistência, acionar Plano B do ROADMAP (HGT no lugar de HeteroSAGE / Transformer no lugar de GRU) — fora do escopo desta fase.

---

# Revisão R1 (2026-06-23) — injeção de popularidade defasada

## Resultado da v1 e causa-raiz

A v1 (T1–T15) rodou o grid completo (24 configs) via `notebooks/phase2_pipeline_treino.ipynb`.
**Nenhuma config supera a persistência ingênua** `ŷ(w)=y(w-1)`:

| Métrica (melhor da grid `W12_h128_l3_lr5e-04`) | GNN val_mse | Persistência val_mse | veredito |
|---|---|---|---|
| combinado (viral50+top200) | ~0.00506 | ~0.0009 | ✗ ~5× pior |

Detalhe no `summary.md` (rodado numa config fraca, `W4_h64_l2_lr1e-03`, 16ª/24): GNN perde em
val **e** test, forecasting **e** retroativo, nos dois regimes.

**Causa-raiz (estrutural, não de hiperparâmetro):** o modelo **nunca recebe o alvo defasado
`y(w-1)`**. A entrada da GRU é puramente o embedding estrutural `bank[wk][song_idxs]`
([`diffusion_gnn.py` `predict`](../../../src/music_diffusion_gnn/models/diffusion_gnn.py)); o
`Sample` carrega `window_weeks` mas **nenhum valor de popularidade**. As features de nó música
são acústicas **estáticas**; só a máscara de arestas muda por semana. Logo o modelo não consegue
fazer o que a persistência faz trivialmente (copiar a última semana) → erra o **nível** da série.

> **Plano B do ROADMAP (HGT/Transformer) NÃO resolve isto** — trocar encoder/cabeça não dá
> acesso ao histórico de popularidade que falta. É problema de **feature/entrada**, não de capacidade.
> O Plano B fica reservado para depois de R1, se ainda houver gap vs. persistência.

## Decisão R1 (gray areas resolvidas)

| # | Decisão | Escolha | Justificativa |
|---|---------|---------|---------------|
| R1-D1 | Onde injetar `y(w-1..w-W)` | **Feature de nó dinâmica** (2 canais: viral50, top200) por semana, antes do HeteroSAGE | Faz a popularidade **difundir pela rede de influência** (hipótese central do trabalho); a estrutura passa a ser load-bearing |
| R1-D2 | Garantir competir com persistência | **Parametrização residual:** `ŷ(w) = clamp(y(w-1) + Δ, 0, 0.5)`, Δ = saída GRU+MLP | Com Δ=0 o modelo **reproduz a persistência exatamente**; só precisa aprender a *correção*. Testa diretamente "estrutura melhora ALÉM do termo AR" |
| R1-D3 | Init para começar na persistência | **Zero-init da última `Linear`** da cabeça (weight=bias=0) → Δ=0 no passo 0 | Inductive bias forte; treino começa empatando a persistência e melhora a partir daí |
| R1-D4 | Fonte de `y(w-1)` do resíduo | Lido do **`pop_bank[w-1, song, chart]`** (mesmo banco da feature) | Banco vem do `weekly_df` **completo** → casa exatamente com a persistência (incl. piso 0.0 em semanas-buraco); **não** muda `build_samples` |

## Mudança de arquitetura

```
pop_bank[w] ∈ (N_music, 2)   # y_viral50(w), y_top200(w); 0 onde fora do chart
node_feat(w) = concat( music.x estático , pop_bank[w] )   # (N_music, d+2)
   → HeteroSAGE (difusão de popularidade pela rede de influência)
   → Z_music(w)
GRU( Z_music[w-W .. w-1] ) → MLP(zero-init) → Δ ∈ ℝ
y_prev = pop_bank[w-1, song, chart]                       # = valor da persistência
ŷ(w)   = clamp( y_prev + Δ , 0 , 0.5 )
```

**Sem leakage:** a feature de popularidade na semana `w'` da janela usa `pop_bank[w']` com
`w' ≤ w-1 < w`; `y_prev` usa `w-1 < w`. Ambos são passado. Coerente com `mask_until(g, w')`.

## Componentes alterados (diff R1)

| Arquivo | Mudança |
|---------|---------|
| `training/dataset.py` | **+** `build_pop_bank(weekly_df, node_id_map_path, n_music, n_weeks=261) -> Tensor (n_weeks, n_music, 2)`. `Sample`/`build_samples` **inalterados**. |
| `models/temporal_head.py` | `forward` retorna **Δ cru** (remove `0.5*sigmoid`); **zero-init** da última `Linear`. |
| `models/diffusion_gnn.py` | `__init__(..., pop_bank=None)` registra buffer; `encode_weeks` concatena `pop_bank[w]` às features de música; `predict` calcula `Δ` e retorna `clamp(y_prev + Δ, 0, 0.5)`. `pop_bank=None` → comportamento de fallback (resíduo base 0) p/ testes. |
| `training/trainer.py` | `pop_bank` opcional em `train_one`, `run_grid`, `evaluate` → repassado ao construtor. |
| `tests/test_phase2_*.py` | range continua via `clamp` (C3); **+** teste: com `pop_bank`, modelo no init ≈ persistência (Δ≈0). |
| `notebooks/phase2_pipeline_treino.ipynb` | constrói `pop_bank` 1×; passa a `train_one`/`run_grid`/`evaluate`/`MusicDiffusionGNN`. |

**Critério de sucesso R1:** reaproveita C6/C7 (GNN < persistência no val MSE, ambos os regimes).
Smoke obrigatório antes do grid: subset pequeno, poucas épocas → GNN deve **empatar ou superar**
persistência (prova que o resíduo funciona). Se empatar mas não superar no grid completo →
estrutura não agrega além do AR; aí sim avaliar Plano B / re-enquadrar (registrar em STATE).
