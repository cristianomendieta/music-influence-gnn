# Phase 7 — Arquitetura com atenção: Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Spec**: `.specs/features/phase-7-atencao/spec.md`
**Pré-compromisso**: `docs/adr/0006-precompromisso-fase-atencao.md` (aceito; nenhuma decisão dele muda depois do primeiro treino)
**Design**: nenhum arquivo separado. O desenho está no ADR-0006 (variante, controles, grade, família do Holm) e nas tasks (assinaturas).
**Status**: Draft (2026-10-18). **Escrita, não implementada.** A implementação fica para a próxima sessão.

**Restrição do autor:** treino, ajuste e avaliação rodam em notebooks no Colab. Localmente só rodam testes unitários leves (tabelas pequenas, grafo sintético de dezenas de nós). Nenhuma task executa um notebook, treina um modelo ou carrega um grafo real aqui.

---

## Antes de começar (decisão do autor, na abertura da sessão de implementação)

Os testes do encoder (T1 a T5) precisam de `torch` e `torch-geometric`, que o venv leve desta máquina não tem. Duas saídas, e a escolha muda só onde o gate Quick roda:

1. **Instalar torch CPU + PyG no venv do scratchpad** (cerca de 300 MB, uma vez). Os testes usam um grafo sintético de dezenas de nós e rodam em segundos. Recomendado: o ciclo de teste fica local.
2. **Rodar a suíte no Colab** numa célula do notebook (como a célula 10 de `item06_08_escada_treinos_colab.ipynb`). Sem instalar nada aqui, mas cada task precisa de uma ida ao Colab.

As tasks T6 a T8 e T13 só usam pandas e numpy e rodam local em qualquer caso.

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `pyproject.toml` (`pythonpath = ["src"]`, marker `slow`). Sem `AGENTS.md` nem `CONTRIBUTING.md`: strong defaults applied. Amostra: `tests/test_graph_controls.py`, `tests/test_phase2_forward.py`, `tests/test_phase2_leakage.py`, `tests/test_phase3_stats.py`, `tests/test_mapa.py` (pytest simples; os de grafo usam o grafo real e dão `skip` se ele falta, os novos usam grafo sintético).

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Encoder, atributos de aresta e integração no modelo (torch) | unit | Todos os ramos; 1:1 com ATT-01 e ATT-02; cada edge case da spec (relação vazia, arestas paralelas, perda NaN fica no notebook) tem teste; grafo **sintético** | `tests/test_phase7_*.py` | `pytest tests/test_phase7_edge_features.py tests/test_phase7_encoder.py tests/test_phase7_model.py -q` |
| Config, treino e orçamento de parâmetros (torch) | unit | Metadados do checkpoint, aborto fora de ±10%, `Config.__str__` | `tests/test_phase7_train.py` | `pytest tests/test_phase7_train.py -q` |
| Veredito e estatística (pandas/numpy) | unit | Todos os ramos; 1:1 com ATT-05 e ATT-06; desfechos O1 a O4, discordância entre splits, família de 18 | `tests/test_veredito_atencao.py`, `tests/test_attention_stats.py` | `pytest tests/test_veredito_atencao.py tests/test_attention_stats.py -q` |
| Regressão do que já existe (torch) | unit | A suíte atual com o encoder padrão `sage` continua verde, com as mesmas chaves de `state_dict` | `tests/test_phase2_*.py`, `tests/test_graph_controls.py`, `tests/test_phase3_*.py`, `tests/test_node_feature_leakage.py` | `pytest tests -q -m "not slow"` |
| Notebooks Colab (`notebooks/item15*.ipynb`) | none | - (gate de sintaxe: JSON válido, células compiláveis, sem saída gravada; execução só no Colab) | `notebooks/item15*.ipynb` | `python scripts/check_notebook.py <nb> --forbid` |
| Docs / specs | none | - | `.specs/**`, `docs/**` | gate de build |

## Gate Check Commands

> Generated from codebase - confirm before Execute. `$PY` é o Python do venv do scratchpad (pandas, scipy, pytest, ruff, nbformat; mais torch CPU e PyG se a opção 1 acima for escolhida).

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | Tasks com testes unitários | o comando `pytest` do arquivo de teste da task, com `$PY -m` na frente |
| Full | Tasks que tocam `diffusion_gnn.py` ou `trainer.py` | Quick mais `$PY -m pytest tests -q -m "not slow"` (regressão com `sage`) |
| Build | Última task de cada fase e tasks de notebook | Full da fase mais `$PY -m ruff check src/music_diffusion_gnn tests` e, nos notebooks, `$PY scripts/check_notebook.py <nb> --forbid` |

---

## Execution Plan

### Phase 1: Encoder de atenção causal (torch)

```
T1 → T2 → T3 → T4 → T5
```

### Phase 2: Veredito do ADR-0006 (pandas, local)

```
T6 → T7 → T8
```

### Phase 3: Notebooks de ajuste, treino, avaliação e veredito (Colab)

```
T5 → T9 → T10 → T11 → T12
```

### Phase 4: Pesos de atenção (P2)

```
T3 → T13 → T14
```

Quatorze tasks: passa de uma levada (~7). Na abertura do Execute, empacotar fases inteiras em levas e **oferecer sub-agentes**; a proposta é Phase 1 numa levada (a mais ambígua, núcleo do experimento), Phases 2 e 3 em outra, Phase 4 em outra. A Phase 2 não depende da 1 (T6 não tem dependência) e pode ir antes se o torch não estiver pronto.

O P3 (ablação por tipo de aresta no `A`, ATT-07) **não entra**: depende do desfecho. Só se abre com O1 ou O2.

---

## Task Breakdown

### T1: Atributos de aresta causais

**What**: `causal_edge_features(snap, week)` devolve, por tipo de aresta, o tensor `(E, 2)` com `[chart, log1p(week − first_seen_week)]` para `(music, cotrajectory, music)` e nada para os outros tipos.
**Where**: `src/music_diffusion_gnn/graph/edge_features.py`
**Depends on**: None
**Reuses**: `graph/temporal.py::mask_until` (a coluna `first_seen_week` é `edge_attr[:, -1]`), layout `[days_together, avg_position_distance, chart, first_seen_week]` de `graph/edges.py::build_cotrajectory`
**Requirement**: ATT-01

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] A saída tem exatamente duas colunas: `chart` (0/1, copiado da coluna 2) e `log1p(week − first_seen_week)`; os valores batem com uma tabela de 5 arestas calculada à mão
- [x] Perturbar as colunas 0 e 1 de `edge_attr` (`days_together`, `avg_position_distance`) para valores arbitrários não altera a saída (AC3 da spec, nível função)
- [x] IF alguma aresta tem `first_seen_week > week` THEN a função lança `ValueError` (snapshot sem `mask_until` é erro de uso, não silêncio)
- [x] Relações sem atributo causal (`performs`, `has_genre`, `rev_has_genre`, `cooccurs`) não aparecem no resultado
- [x] A idade nunca é negativa em arestas de um snapshot mascarado
- [x] Gate check passes: `$PY -m pytest tests/test_phase7_edge_features.py -q`
- [x] Test count: ≥ 5 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(graph): derive causal edge features from the snapshot week`

---

### T2: Encoder GATv2 por relação com termo raiz

**What**: `HeteroAttentionEncoder(metadata, hidden, layers, heads, dropout, attn_dropout)` aplica, por camada, `ReLU(W_raiz,t·x + Σ_r GATv2_r(x_origem, x_destino))`, com `add_self_loops=False`, `edge_dim=2` só na cotrajetória e saída `Z_music`.
**Where**: `src/music_diffusion_gnn/models/attention_encoder.py`
**Depends on**: T1
**Reuses**: `models/encoder.py::HeteroSpatialEncoder` (mesma interface `forward(x_dict, edge_index_dict)`, mesmo `ReLU` e dropout entre camadas), `GATv2Conv` do PyG (`in_channels=(-1,-1)`, `concat=True`, canais por cabeça = `hidden / heads`)
**Requirement**: ATT-01, ATT-02

**Tools**:

- MCP: `context7` (confirmar a assinatura de `GATv2Conv` na versão do PyG instalada, se houver dúvida)
- Skill: NONE

**Done when**:

- [x] A saída tem forma `(N_music, hidden)` num grafo sintético com os cinco tipos de aresta
- [x] IF uma relação não tem aresta no snapshot THEN o encoder não lança erro e o tipo de nó destino continua com embedding não nulo, vindo do termo raiz (AC7)
- [x] Perturbar as colunas brutas `days_together` e `avg_position_distance` do `edge_attr` do grafo devolve embeddings idênticos (AC3, nível encoder)
- [x] Paridade funcional (AC4): com todas as arestas removidas (`strip_all_edges`) e com `W_raiz,t` e `b_raiz,t` iguais à soma dos `lin_r` e dos vieses de `lin_l` das relações do `HeteroSpatialEncoder` que chegam ao tipo `t`, o embedding de música é igual ao do SAGE (tolerância `1e-6`)
- [x] `hidden` não divisível por `heads` lança `ValueError` na construção, com mensagem que diz os dois números
- [x] Gate check passes: `$PY -m pytest tests/test_phase7_encoder.py -q`
- [x] Test count: ≥ 6 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(models): add per-relation GATv2 encoder with a root term`

---

### T3: Pesos de atenção e orçamento de parâmetros do encoder

**What**: `HeteroAttentionEncoder.forward(..., return_attention=True)` devolve também, por relação e por camada, `(edge_index, alpha)`; `count_params()` conta só parâmetros já inicializados, como em `MusicDiffusionGNN.count_params`.
**Where**: `src/music_diffusion_gnn/models/attention_encoder.py` (modify)
**Depends on**: T2
**Reuses**: `return_attention_weights` do `GATv2Conv`, `MusicDiffusionGNN.count_params`
**Requirement**: ATT-02

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Por relação e camada, os pesos somam 1 sobre as arestas de entrada de cada nó de destino com ao menos uma aresta (AC5); cabeças somadas separadamente, tolerância `1e-5`
- [x] Arestas paralelas (um par nos dois charts) aparecem como duas entradas distintas, cada uma com seu peso (edge case da spec)
- [x] Com `return_attention=False` (o padrão) a saída é idêntica à da T2: o caminho padrão não muda
- [x] `count_params()` depois de um forward devolve um inteiro positivo; antes do forward não conta os parâmetros lazy
- [x] Gate check passes: `$PY -m pytest tests/test_phase7_encoder.py -q`
- [x] Test count: ≥ 10 tests pass in total (no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(models): expose attention weights and parameter count of the encoder`

---

### T4: Escolher o encoder dentro do `MusicDiffusionGNN`

**What**: `MusicDiffusionGNN(encoder="sage"|"gatv2", heads=4, attn_dropout=0.0)`; `encode_weeks` monta os atributos de aresta com `causal_edge_features(snap, w)` e os entrega ao encoder de atenção. O padrão `sage` fica byte a byte como está.
**Where**: `src/music_diffusion_gnn/models/diffusion_gnn.py` (modify)
**Depends on**: T3
**Reuses**: `graph/edge_features.py`, `models/attention_encoder.py`, `_subsample_cotraj` (já subamostra `edge_attr`)
**Requirement**: ATT-01

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Com `encoder="sage"` as chaves e as formas do `state_dict` são idênticas às de antes da mudança (teste contra uma lista de chaves fixada no teste) e a regressão existente passa
- [x] Com `encoder="gatv2"`, `encode_weeks` e `predict` rodam num grafo sintético e devolvem predições em `[0, 0,5]`
- [x] Perturbar `days_together` e `avg_position_distance` no grafo não altera as predições de ponta a ponta (AC3 inteiro)
- [x] Com `max_cotraj_edges` o atributo de aresta é subamostrado junto com as arestas (mesmo `perm`)
- [x] `encoder` desconhecido lança `ValueError` que lista os valores aceitos
- [x] Gate check passes: `$PY -m pytest tests/test_phase7_model.py -q` e a regressão `$PY -m pytest tests -q -m "not slow"`
- [x] Test count: ≥ 5 new tests pass and the pre-existing suite count is unchanged (no silent deletions)

**Tests**: unit
**Gate**: full

**Commit**: `feat(models): select the sage or gatv2 encoder in MusicDiffusionGNN`

---

### T5: Config, treino e checkpoint com metadados e aborto fora do orçamento

**What**: `Config` ganha `encoder`, `heads`, `attn_dropout`; `train_one` os repassa ao modelo; `check_param_budget(n_params, ref=453_121, tol=0.10)` aborta o treino fora da faixa; o checkpoint grava encoder, heads, atenção, variante e versão do grafo.
**Where**: `src/music_diffusion_gnn/training/trainer.py` (modify)
**Depends on**: T4
**Reuses**: `Config.__str__`, `TrainResult.n_params`, o formato de checkpoint de `item06_08_escada_treinos_colab.ipynb` (célula 14)
**Requirement**: ATT-02, ATT-04

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `Config.__str__` distingue encoders e cabeças (`..._gatv2_h4_ad0.2`), e para `sage` continua igual ao texto de hoje (`W12_h128_l3_lr5e-04`)
- [x] `check_param_budget` aceita `453_121 ± 10%` e lança erro com os dois números fora disso (limites 407.808 e 498.433 testados)
- [x] O dicionário de checkpoint inclui `encoder`, `heads`, `attn_dropout`, `variante`, `split_regime`, `seed`, `graph_seed` e `n_params`, e um modelo é reconstruído a partir só desses campos (AC4 de treino)
- [x] `train_one` roda 1 época num grafo e amostras sintéticas com `encoder="gatv2"` e devolve `n_params` e `val_mse` finitos
- [x] Gate check passes: `$PY -m pytest tests/test_phase7_train.py -q` e a regressão `$PY -m pytest tests -q -m "not slow"`
- [x] Test count: ≥ 5 new tests pass and the pre-existing suite count is unchanged (no silent deletions)

**Tests**: unit
**Gate**: build

**Commit**: `feat(training): carry encoder settings in Config and enforce the parameter budget`

---

### T6: Comparação dos sete modelos em 18 testes por split

**What**: `compare_models(preds, *, B, seed)` calcula H1 (`A` contra sem_grafo), H2 (`A` contra `A-emb`) e H3 (`A` contra a proposta SAGE) por split e por célula (chart × k) on-chart, com Holm sobre as 18 comparações de cada split.
**Where**: `src/music_diffusion_gnn/evaluation/veredito_atencao.py`
**Depends on**: None
**Reuses**: `evaluation/stats.py::paired_by_song`, `holm_correction`, `evaluation/mapa.py::label_cell` (mesma regra de vence/perde)
**Requirement**: ATT-05

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] A tabela tem 6 células × 3 contrastes = 18 linhas por split, e o `p_holm` de cada split é `holm_correction` aplicada às 18 `p_value` juntas
- [ ] "Vence" é IC inteiro abaixo de zero **e** `p_holm < 0,05`; "perde" é o espelho; só on-chart entra
- [ ] A diferença de cada linha coincide com `paired_by_song` chamada diretamente (mesma semente)
- [ ] Dentro de uma célula (chart × k) as origens da mesma música são promediadas antes do teste, e `n_songs` conta músicas, não origens
- [ ] Um modelo ausente nas predições lança `ValueError` que nomeia o modelo
- [ ] Gate check passes: `$PY -m pytest tests/test_veredito_atencao.py -q`
- [ ] Test count: ≥ 6 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(veredito): test the attention encoder against its three references`

---

### T7: Classificar o desfecho O1 a O4 por split

**What**: `classify_outcome(table)` devolve, por split, `O4` se `A` perde para o sem_grafo em ≥ 4 de 6 células; senão `O1` se H1 e H2 vencem em ≥ 4; senão `O2` se só H1; senão `O3`; e `discordancia = True` quando os dois splits diferem.
**Where**: `src/music_diffusion_gnn/evaluation/veredito_atencao.py` (modify)
**Depends on**: T6
**Reuses**: T6
**Requirement**: ATT-05

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Quatro tabelas sintéticas produzem exatamente O1, O2, O3 e O4 (uma por desfecho), e 3 de 6 vitórias **não** conta como maioria enquanto 4 de 6 conta
- [ ] H2 sem H1 produz `O3` (cláusula fechada no ADR-0006 em 2026-10-18)
- [ ] `O4` tem precedência: uma tabela com H2 em maioria e `A` perdendo para o sem_grafo em ≥ 4 devolve `O4`
- [ ] H3 não altera o desfecho: duas tabelas iguais exceto por H3 devolvem o mesmo código
- [ ] Splits com desfechos diferentes marcam `discordancia = True` e a saída diz que a afirmação principal se restringe ao que se replica
- [ ] Gate check passes: `$PY -m pytest tests/test_veredito_atencao.py -q`
- [ ] Test count: ≥ 12 tests pass in total (no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(veredito): classify the ADR-0006 outcome per split`

---

### T8: Sensibilidade por seed isolada e bloco de orçamento

**What**: `sensitivity_by_seed(preds_seeds, ...)` repete H1 a H3 em cada seed e conta quantas concordam com o teste sobre a média; `budget_block(...)` monta o texto com parâmetros e orçamento de ajuste (8 contra 24 configurações).
**Where**: `src/music_diffusion_gnn/evaluation/veredito_atencao.py` (modify)
**Depends on**: T7
**Reuses**: T6, T7, o padrão de `evaluation/mapa.py::sensitivity_by_seed`
**Requirement**: ATT-05

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Com 3 seeds iguais a concordância é 3 de 3 em todas as células; com uma seed invertida é 2 de 3
- [ ] O resultado traz o desfecho O1 a O4 por seed isolada ao lado do desfecho da média
- [ ] `budget_block` inclui o número de parâmetros do `A`, o da proposta SAGE (453.121) e "8 contra 24 configurações" (AC6)
- [ ] Gate check passes: `$PY -m pytest tests/test_veredito_atencao.py -q`
- [ ] Test count: ≥ 15 tests pass in total (no silent deletions)

**Tests**: unit
**Gate**: build

**Commit**: `feat(veredito): add per-seed sensitivity and the budget block`

---

### T9: Notebook do ajuste das 8 configurações

**What**: `notebooks/item15a_ajuste_atencao_colab.ipynb` treina as 8 configurações (`heads` {2,4} × `attn_dropout` {0, 0,2} × `lr` {5e-4, 1e-3}, `hidden=128`) em `current`/seed 42, grava `val_mse` e parâmetros de cada uma antes de qualquer avaliação em teste, escolhe a de menor `val_mse` (empate dentro da dispersão: a de menos parâmetros) e grava `config_escolhida.json`.
**Where**: `notebooks/item15a_ajuste_atencao_colab.ipynb`
**Depends on**: T5
**Reuses**: células de setup, de dados e de retomada de `notebooks/item06_08_escada_treinos_colab.ipynb`; `Config`, `train_one`, `check_param_budget`
**Requirement**: ATT-03

**Tools**:

- MCP: `colab-mcp` (só se o autor pedir para abrir o notebook no Colab)
- Skill: NONE

**Done when**:

- [ ] O notebook roda a suíte `tests/test_phase7_*.py` numa célula inicial **como conferência** (como a célula 10 do item 06), e o passo seguinte só usa grafo real depois dessa célula
- [ ] Um teste de invariância às colunas brutas de `edge_attr` roda no **grafo real** antes do primeiro treino e aborta o notebook se falhar (portão de vazamento do ADR-0006)
- [ ] Cada configuração é retomável por arquivo no Drive, e uma sessão nova pula as que existem
- [ ] A tabela de 8 linhas (`val_mse`, `n_params`) é gravada antes da escolha, e `check_param_budget` roda em todas
- [ ] O notebook aplica a regra de empate (diferença de `val_mse` ≤ 0,000003, um desvio entre seeds medido no item 06) escolhendo a configuração de menos parâmetros, e registra quando ela decide
- [ ] O notebook não tem saída gravada e compila
- [ ] Gate check passes: `$PY scripts/check_notebook.py notebooks/item15a_ajuste_atencao_colab.ipynb --forbid`
- [ ] Test count: all unit tests of T1 to T8 still pass (no silent deletions)

**Tests**: none
**Gate**: build

**Commit**: `feat(atencao): add Colab notebook for the fixed 8-config tuning`

---

### T10: Notebook dos 12 treinos

**What**: `notebooks/item15b_treinos_atencao_colab.ipynb` treina `A` e `A-emb` × 2 splits × 3 seeds com a configuração escolhida na T9, `max_cotraj_edges = 30_000`, 35 épocas, retomável por checkpoint, com o grafo embaralhado de `rewire_preserving_degree(g, seed=seed)`.
**Where**: `notebooks/item15b_treinos_atencao_colab.ipynb`
**Depends on**: T9
**Reuses**: `notebooks/item06_08_escada_treinos_colab.ipynb` (plano de treinos, retomada, tempo-limite da sessão), `graph/controls.py`
**Requirement**: ATT-03, ATT-04

**Tools**:

- MCP: `colab-mcp` (só se o autor pedir)
- Skill: NONE

**Done when**:

- [ ] O notebook aborta com mensagem clara se `config_escolhida.json` não existir ou se a configuração mudar depois de criada (hash gravado no JSON)
- [ ] O plano lista exatamente 12 treinos (`A`, `A-emb` × `current`, `pre_pandemia` × 42, 43, 44) e cada checkpoint grava regime, seed, variante, configuração, versão do grafo e `graph_seed` do embaralhado
- [ ] O orçamento de parâmetros é impresso e `check_param_budget` roda em cada treino; o notebook compara com a proposta SAGE (453.121) numa célula
- [ ] IF a perda vira NaN numa seed THEN o notebook registra a falha, repete uma vez com outra inicialização e grava as duas tentativas (edge case da spec)
- [ ] O notebook não tem saída gravada e compila
- [ ] Gate check passes: `$PY scripts/check_notebook.py notebooks/item15b_treinos_atencao_colab.ipynb --forbid`
- [ ] Test count: all unit tests of T1 to T8 still pass (no silent deletions)

**Tests**: none
**Gate**: build

**Commit**: `feat(atencao): add Colab notebook for the 12 attention trainings`

---

### T11: Notebook de avaliação do Modo 2

**What**: `notebooks/item15c_avaliacao_atencao_colab.ipynb` roda o Modo 2 (k ∈ {1,2,4}, dois charts, dois recortes) para `A` e `A-emb` nas duas leituras de orçamento de arestas, com o mesmo esquema de colunas dos `mode2_*.parquet` do item 09.
**Where**: `notebooks/item15c_avaliacao_atencao_colab.ipynb`
**Depends on**: T10
**Reuses**: `notebooks/item09_avaliacao_matriz_colab.ipynb` (origens por regime, `marcar_recorte`, retomada), `evaluation/rollout.py::gnn_rollout_recursive`
**Requirement**: ATT-04

**Tools**:

- MCP: `colab-mcp` (só se o autor pedir)
- Skill: NONE

**Done when**:

- [ ] Os `mode2_*_completo.parquet` e `mode2_*_teto_treino.parquet` têm as colunas `variante`, `split_regime`, `seed`, `song_id`, `chart`, `origin_week`, `k`, `y_true`, `y_pred`, `onchart`, `orcamento_arestas`, e `variante` vale `atencao` ou `atencao_emb`
- [ ] As origens são exatamente as de `origens_{regime}.parquet` do item 09 (conferência de contagem: 141.504 no `current` e 53.982 no `pre_pandemia` por modelo, somando os horizontes)
- [ ] O notebook carrega cada checkpoint pelos metadados gravados (T5), sem constantes de configuração repetidas
- [ ] A leitura `completo` é a principal; a `teto_treino` roda em todas as seeds (o item 09 a fez só na seed 42) ou o notebook declara que só rodou em uma
- [ ] O notebook não tem saída gravada e compila
- [ ] Gate check passes: `$PY scripts/check_notebook.py notebooks/item15c_avaliacao_atencao_colab.ipynb --forbid`
- [ ] Test count: all unit tests of T1 to T8 still pass (no silent deletions)

**Tests**: none
**Gate**: build

**Commit**: `feat(atencao): add Colab notebook that runs Mode 2 for the attention models`

---

### T12: Notebook do veredito

**What**: `notebooks/item15d_veredito_atencao_colab.ipynb` junta `predicoes_cinco_modelos.parquet` (item 10) com as predições de `A` e `A-emb` (média das seeds, como no item 10), roda `compare_models`, `classify_outcome`, a sensibilidade por seed e escreve o relatório com a tabela de desfecho.
**Where**: `notebooks/item15d_veredito_atencao_colab.ipynb`
**Depends on**: T11
**Reuses**: `notebooks/item10_comparacao_consolidada.ipynb` (agregação por seed, tabelas), `evaluation/veredito_atencao.py` (T6 a T8)
**Requirement**: ATT-05

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] A primeira célula de texto reproduz a tabela O1 a O4 e a regra de maioria do ADR-0006 **antes** de qualquer célula que leia resultado
- [ ] O desfecho sai por split, com a discordância destacada quando existir, e cada seed isolada aparece na tabela de sensibilidade
- [ ] O relatório traz parâmetros do `A` contra 453.121 e "8 contra 24 configurações" como limitação
- [ ] O notebook não recalcula nada que o ADR fixou (limiares, família, contrastes são importados do módulo, não digitados)
- [ ] O notebook escreve `results/atencao/veredito.parquet` e `relatorio.md` e não tem saída gravada
- [ ] Gate check passes: `$PY scripts/check_notebook.py notebooks/item15d_veredito_atencao_colab.ipynb --forbid`
- [ ] Test count: all unit tests of T1 to T8 still pass (no silent deletions)

**Tests**: none
**Gate**: build

**Commit**: `feat(atencao): add Colab notebook with the ADR-0006 verdict`

---

### T13: Entropia da atenção por grau de entrada

**What**: `normalized_attention_entropy(edge_index, alpha, n_dst)` devolve a entropia normalizada da distribuição de atenção por nó de destino, e `entropy_by_degree(...)` a agrega por faixa de grau de entrada, para `A` e `A-emb`.
**Where**: `src/music_diffusion_gnn/evaluation/attention_stats.py`
**Depends on**: T3
**Reuses**: a saída de `return_attention=True` da T3
**Requirement**: ATT-06

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Atenção uniforme em `d` vizinhos dá entropia normalizada 1; atenção concentrada num só vizinho dá 0; grau 1 dá 0 por convenção documentada (log de 1 não normaliza)
- [ ] A entropia é invariante à ordem das arestas de entrada
- [ ] `entropy_by_degree` devolve uma linha por faixa de grau com a contagem de nós, e a soma das contagens é o número de nós com ao menos uma aresta
- [ ] Gate check passes: `$PY -m pytest tests/test_attention_stats.py -q`
- [ ] Test count: ≥ 5 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(atencao): measure the entropy of attention by in-degree`

---

### T14: Notebook dos pesos de atenção

**What**: `notebooks/item15e_pesos_atencao_colab.ipynb` extrai os pesos on-chart de `A` e `A-emb`, plota a entropia por grau de entrada e, se o mapa da Phase 6b tiver o estrato por grau, cruza o ganho do `A` sobre o SAGE com o tercil de grau.
**Where**: `notebooks/item15e_pesos_atencao_colab.ipynb`
**Depends on**: T13
**Reuses**: `evaluation/attention_stats.py`, `notebooks/item15c_avaliacao_atencao_colab.ipynb` (carga de checkpoints)
**Requirement**: ATT-06

**Tools**:

- MCP: NONE
- Skill: `dataviz` (figuras com entropia por grau)

**Done when**:

- [ ] Toda figura traz a legenda "descritiva: atenção não é explicação" e remete à ablação por tipo de aresta como medida causal
- [ ] O cruzamento com o tercil de grau só roda se o arquivo do estrato por grau (Phase 6b, MAP-08) existir, e diz que pulou quando não existe
- [ ] Os pesos vêm de `A` e `A-emb` com os mesmos nós e semanas, para comparação direta
- [ ] O notebook não tem saída gravada e compila
- [ ] Gate check passes: `$PY scripts/check_notebook.py notebooks/item15e_pesos_atencao_colab.ipynb --forbid`
- [ ] Test count: all unit tests of T1 to T13 still pass (no silent deletions)

**Tests**: none
**Gate**: build

**Commit**: `feat(atencao): add Colab notebook that describes the attention weights`

---

## Phase Execution Map

```
Phase 1:  T1 → T2 → T3 → T4 → T5
Phase 2:  T6 → T7 → T8
Phase 3:  T5 → T9 → T10 → T11 → T12
Phase 4:  T3 → T13 → T14
```

Execução sequencial. As Phases 1 e 2 são independentes entre si e a 2 não precisa de torch. A Phase 3 só começa depois da T5 e da T8 (o veredito da T12 importa o módulo da Phase 2). A Phase 4 pode ir a qualquer momento depois da T3, mas é a menos prioritária (P2).

**Prazo (ADR-0006):** matriz concluída até 2026-11-27. Com 14 tasks, o caminho crítico é T1 → T5 → T9 → T10 → T11 → T12. Cada treino leva tempo desconhecido para a atenção (a referência do SAGE é ~7 min); medir na T9 e recalcular o prazo antes de lançar a T10.

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: atributos de aresta | 1 função | ✅ Granular |
| T2: encoder | 1 classe | ✅ Granular |
| T3: pesos e orçamento | 1 método e 1 contador, mesma classe | ⚠️ OK, coesas |
| T4: encoder no modelo | 1 mudança em 1 arquivo | ✅ Granular |
| T5: Config, treino, checkpoint | 3 mudanças pequenas em 1 arquivo | ⚠️ OK, coesas |
| T6: compare_models | 1 função | ✅ Granular |
| T7: classify_outcome | 1 função | ✅ Granular |
| T8: sensibilidade e bloco | 2 funções, mesmo arquivo | ⚠️ OK, coesas |
| T9 a T12, T14: um notebook cada | 1 notebook | ✅ Granular |
| T13: entropia | 2 funções, mesmo arquivo | ⚠️ OK, coesas |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | início da Phase 1 | ✅ Match |
| T2 | T1 | T1 → T2 | ✅ Match |
| T3 | T2 | T2 → T3 | ✅ Match |
| T4 | T3 | T3 → T4 | ✅ Match |
| T5 | T4 | T4 → T5 | ✅ Match |
| T6 | None | início da Phase 2 | ✅ Match |
| T7 | T6 | T6 → T7 | ✅ Match |
| T8 | T7 | T7 → T8 | ✅ Match |
| T9 | T5 | T5 → T9 | ✅ Match |
| T10 | T9 | T9 → T10 | ✅ Match |
| T11 | T10 | T10 → T11 | ✅ Match |
| T12 | T11 | T11 → T12 | ✅ Match |
| T13 | T3 | T3 → T13 | ✅ Match |
| T14 | T13 | T13 → T14 | ✅ Match |

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | atributos de aresta (torch) | unit | unit | ✅ OK |
| T2 | encoder (torch) | unit | unit | ✅ OK |
| T3 | encoder (torch) | unit | unit | ✅ OK |
| T4 | modelo (torch) | unit | unit | ✅ OK |
| T5 | trainer (torch) | unit | unit | ✅ OK |
| T6 | veredito (pandas) | unit | unit | ✅ OK |
| T7 | veredito (pandas) | unit | unit | ✅ OK |
| T8 | veredito (pandas) | unit | unit | ✅ OK |
| T9 | notebook Colab | none | none | ✅ OK |
| T10 | notebook Colab | none | none | ✅ OK |
| T11 | notebook Colab | none | none | ✅ OK |
| T12 | notebook Colab | none | none | ✅ OK |
| T13 | estatística da atenção (numpy) | unit | unit | ✅ OK |
| T14 | notebook Colab | none | none | ✅ OK |
