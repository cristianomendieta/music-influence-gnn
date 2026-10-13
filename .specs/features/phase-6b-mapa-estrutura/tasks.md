# Phase 6b — Mapa de onde a estrutura paga: Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Spec**: `.specs/features/phase-6b-mapa-estrutura/spec.md`
**Design**: nenhum. Feature pequena de funções puras sobre pandas; o desenho está inline nas tasks.
**Status**: Approved (2026-10-13)

**Restrição do autor:** a execução pesada roda no Colab. Localmente só rodam os testes unitários (pandas, segundos). Nenhuma task executa o notebook, o bootstrap de 10.000 reamostras ou qualquer carga do grafo aqui.

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `pyproject.toml` (`[tool.pytest.ini_options]`, `pythonpath = ["src"]`, marker `slow`). Sem `AGENTS.md` nem `CONTRIBUTING.md`: strong defaults applied. Amostra: `tests/test_phase3_stats.py`, `tests/test_phase3_longhits.py`, `tests/test_split_regime.py` (pytest simples, funções puras, dados sintéticos).

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Funções puras de avaliação (`evaluation/strata.py`, `evaluation/mapa.py`) | unit | Todos os ramos; 1:1 com os ACs MAP-01 a MAP-07; cada edge case da spec tem teste | `tests/test_strata.py`, `tests/test_mapa.py` | `pytest tests/test_strata.py tests/test_mapa.py -q` |
| Notebook Colab (`notebooks/item22_mapa_estrutura_colab.ipynb`) | none | - (gate de sintaxe: JSON válido e células de código compiláveis; execução só no Colab) | `notebooks/*.ipynb` | gate de sintaxe do notebook (abaixo) |
| Docs / specs | none | - | `.specs/**`, `docs/**` | gate de build |

## Gate Check Commands

> Generated from codebase - confirm before Execute. Rodam no venv leve do scratchpad (pandas, scipy, numpy, pytest, ruff, nbformat); o ambiente do projeto não tem torch aqui e estas tasks não precisam dele.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | Tasks com testes unitários | `$PY -m pytest tests/test_strata.py tests/test_mapa.py -q` |
| Full | Sem integração além do unit nesta feature | igual ao Quick |
| Build | Última task de cada fase e tasks sem teste | `$PY -m pytest tests/test_strata.py tests/test_mapa.py -q && $PY -m ruff check src/music_diffusion_gnn/evaluation/strata.py src/music_diffusion_gnn/evaluation/mapa.py` e, para o notebook, `$PY scripts/check_notebook.py notebooks/item22_mapa_estrutura_colab.ipynb` |

`$PY` = `/tmp/claude-1000/-home-cristiano-estudos-music-influence-gnn/d471120b-f6bb-4cf1-b07b-79f0e7d48e4a/scratchpad/.venv/bin/python`.

---

## Execution Plan

### Phase 1: Estratos, veredito, sensibilidade e notebook

Uma única fase: as sete tasks formam uma cadeia de dependências (cada uma usa a função da anterior), então não há corte de fase honesto.

```
T1 → T2 → T3 → T4 → T5 → T6 → T7
```

Sete tasks, uma única leva: execução inline, sem sub-agentes. O P3 (grau de cotrajetória, MAP-08) fica fora destas tasks: exige torch e o grafo, só roda no Colab, e depende do mapa principal estar pronto. Entra num `tasks` próprio depois.

---

## Task Breakdown

### T1: Atribuir o estrato de vida a cada origem

**What**: `assign_strata(origins, onchart, *, young_weeks=8, wave_gap=4)` devolve um estrato entre `estreia`, `reentrada`, `jovem`, `maduro_mono`, `maduro_multi` para cada linha de `origins` (`song_id`, `chart`, `origin_week`), usando só semanas `≤ origin_week`.
**Where**: `src/music_diffusion_gnn/evaluation/strata.py`
**Depends on**: None
**Reuses**: `evaluation/longhits.py::onchart_weeks` (formato do conjunto `(song_id, chart, week)`)
**Requirement**: MAP-01, MAP-02

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Cada origem recebe exatamente um dos cinco estratos; a tabela verdade da spec (estreia, reentrada, jovem, maduro_mono, maduro_multi) é reproduzida em séries sintéticas de estágio conhecido
- [x] Os limites estão testados: idade 8 é `jovem`, idade 9 é `maduro_*`; hiato de 3 semanas fora do chart não abre onda, hiato de 4 abre
- [x] Teste de truncamento: o estrato com a série completa é idêntico ao estrato com a série cortada em `origin_week`, para todas as origens da série sintética (MAP-02, AC3 da spec)
- [x] Origem anterior à primeira semana on-chart da série cai em `estreia`
- [x] Gate check passes: `$PY -m pytest tests/test_strata.py -q`
- [x] Test count: 17 tests pass (≥ 8 planned; no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(strata): assign causal life-stage stratum to each forecast origin`

---

### T2: Marcar `sem_serie` e tabular contagens com o corte de 30 músicas

**What**: `assign_strata` passa a aceitar `series_keys` e marca `sem_serie` para (música, chart) ausentes; `strata_counts(origins_with_strata, *, min_songs=30)` devolve origens e músicas por (split, estrato, k) com a bandeira `insuficiente`.
**Where**: `src/music_diffusion_gnn/evaluation/strata.py` (modify)
**Depends on**: T1
**Reuses**: T1
**Requirement**: MAP-01, MAP-03

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Uma origem cujo (música, chart) não está em `series_keys` recebe `sem_serie` e é contada à parte no relatório de contagens
- [x] `strata_counts` devolve `n_origens` e `n_musicas` corretos numa tabela sintética com contagem conhecida, e `insuficiente = True` exatamente quando `n_musicas < 30` (29 sim, 30 não)
- [x] A soma das origens dos cinco estratos mais `sem_serie` é igual ao total de origens de entrada
- [x] Gate check passes: `$PY -m pytest tests/test_strata.py -q`
- [x] Test count: 24 tests pass in total (≥ 14 planned; no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(strata): flag origins without a series and mark undersized cells`

---

### T3: Comparar duas variantes numa célula

**What**: `compare_cell(err_a, err_b, song_ids, *, B, seed, equiv_frac=0.05)` agrega o erro quadrático por música (música em dois charts conta uma vez) e devolve a diferença média, o IC bootstrap 95%, o `p` do Wilcoxon, a taxa de vitória, o número de músicas e a margem de equivalência.
**Where**: `src/music_diffusion_gnn/evaluation/mapa.py`
**Depends on**: T2
**Reuses**: `evaluation/stats.py::wilcoxon_signed_rank`, `bootstrap_ci_diff`
**Requirement**: MAP-04

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] A diferença média, o IC e o `p` coincidem com a chamada direta a `stats.paired_by_song` em dados sintéticos, e a margem é `equiv_frac × média por música de err_b`
- [ ] Duas origens da mesma música em charts diferentes produzem uma única observação por música
- [ ] Erros idênticos nas duas variantes devolvem `p = 1.0` e diferença 0, sem lançar exceção (o Wilcoxon do scipy falha com diferenças todas nulas)
- [ ] A semente fixa torna o IC reprodutível: duas chamadas iguais devolvem o mesmo IC
- [ ] Gate check passes: `$PY -m pytest tests/test_mapa.py -q`
- [ ] Test count: ≥ 5 tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(mapa): compare two variants per song within a stratum cell`

---

### T4: Rotular a célula e aplicar Holm numa família fixa

**What**: `label_cell(ci_lo, ci_hi, p_holm, margin)` devolve `vence`, `perde`, `equivalente` ou `inconclusivo`; `holm_family(tests)` aplica `holm_correction` a todas as células testadas de um split, nos dois contrastes juntos, deixando fora as `insuficiente`.
**Where**: `src/music_diffusion_gnn/evaluation/mapa.py` (modify)
**Depends on**: T3
**Reuses**: `evaluation/stats.py::holm_correction`
**Requirement**: MAP-05

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `label_cell` devolve `vence` só com IC inteiro abaixo de zero **e** `p_holm < 0,05`; `perde` com IC inteiro acima de zero e `p_holm < 0,05`; `equivalente` com IC dentro de ±margem; `inconclusivo` no resto; cada rótulo tem um caso sintético, e os limites (`p_holm = 0,05`, IC tocando zero) estão cobertos
- [ ] A regra de precedência está testada: IC dentro da margem **e** abaixo de zero com `p_holm` pequeno devolve `vence`, não `equivalente`
- [ ] `holm_family` produz os mesmos `p` ajustados que `holm_correction` aplicada à lista de células testadas, excluindo as `insuficiente`; o tamanho da família é `≤ 30` por split
- [ ] Gate check passes: `$PY -m pytest tests/test_mapa.py -q`
- [ ] Test count: ≥ 12 tests pass in total (no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(mapa): label cells and correct one fixed Holm family per split`

---

### T5: Montar o mapa e o veredito "estrutura paga"

**What**: `build_map(preds, strata, *, B, seed)` produz a tabela (split, estrato, k, contraste, rótulo, IC, `p` Holm, músicas) a partir das predições dos três modelos neurais e dos estratos; `structure_pays(map_df)` marca o estrato `estrutura paga` quando vence os dois controles em ≥ 2 de 3 horizontes nos dois splits.
**Where**: `src/music_diffusion_gnn/evaluation/mapa.py` (modify)
**Depends on**: T4
**Reuses**: T2, T3, T4
**Requirement**: MAP-04, MAP-05, MAP-06

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Num conjunto sintético em que a proposta vence os dois controles num estrato nos dois splits em 2 de 3 horizontes, `structure_pays` marca esse estrato e só ele
- [ ] Vencer em um split e não no outro **não** marca `estrutura paga`; vencer só um dos controles também não
- [ ] Um estrato `insuficiente` em um dos splits fica `não testável` e nunca `estrutura paga`
- [ ] A tabela tem uma linha por (split, estrato, k, contraste) testável, e o rótulo de cada linha é recomputável a partir de IC e `p` Holm da própria linha
- [ ] Uma música presente nos dois charts entra uma vez por célula
- [ ] Gate check passes: `$PY -m pytest tests/test_mapa.py -q`
- [ ] Test count: ≥ 18 tests pass in total (no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(mapa): build the per-stratum map and the structure-pays verdict`

---

### T6: Sensibilidade por seed, por chart e por margem

**What**: `sensitivity_by_seed(preds_seeds, strata, ...)`, `by_chart_rmse(preds, strata)` e `margin_sensitivity(map_df_inputs, margins=(0.025, 0.10))` devolvem, respectivamente, a concordância do rótulo entre as 3 seeds, a diferença de RMSE descritiva por chart e os rótulos que mudam com a margem.
**Where**: `src/music_diffusion_gnn/evaluation/mapa.py` (modify)
**Depends on**: T5
**Reuses**: T5
**Requirement**: MAP-07

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Em dados sintéticos com rótulo estável entre seeds, a concordância é 3/3; com uma seed discordando, 2/3
- [ ] `by_chart_rmse` devolve a diferença de RMSE por (estrato, chart) com os valores corretos de uma tabela conhecida e não produz `p` nem rótulo
- [ ] `margin_sensitivity` lista exatamente as células cujo rótulo muda entre 2,5%, 5% e 10%, e uma célula com IC largo fora de qualquer margem não aparece
- [ ] Gate check passes: `$PY -m pytest tests/test_mapa.py -q`
- [ ] Test count: ≥ 22 tests pass in total (no silent deletions)

**Tests**: unit
**Gate**: quick

**Commit**: `feat(mapa): add seed, chart and margin sensitivity of the map`

---

### T7: Notebook Colab que roda o mapa de ponta a ponta

**What**: `notebooks/item22_mapa_estrutura_colab.ipynb`: cabeçalho com hipótese, estratos e limiares escritos antes dos resultados; setup do Colab; carga de `predicoes_cinco_modelos.parquet` e `timeseries.parquet`; estratos e contagens; mapa; sensibilidades (seed, quando o Drive tem os `mode2_*.parquet`); tabelas e figura; gravação de `results/mapa_estrutura/*.parquet`. Mais o `scripts/check_notebook.py` que serve de gate.
**Where**: `notebooks/item22_mapa_estrutura_colab.ipynb`
**Depends on**: T6
**Reuses**: células de setup de `notebooks/item10_comparacao_consolidada.ipynb`, T1 a T6
**Requirement**: MAP-03, MAP-05, MAP-06, MAP-07

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] O notebook é um JSON válido (nbformat 4) e todas as células de código compilam (`ast.parse`, ignorando magics `%`)
- [ ] A primeira célula de texto traz a hipótese a priori, os cinco estratos, `H = 8`, `wave_gap = 4`, `min_songs = 30` e a margem de 5%, antes de qualquer célula que leia resultado
- [ ] O notebook procura as predições primeiro em `results/comparacao_final/` do clone e depois no Drive, e aborta com mensagem clara se não achar em nenhum
- [ ] O notebook não importa torch nem carrega o grafo, e não tem célula que reconstrua predições
- [ ] O notebook não tem saída gravada (nada executado aqui)
- [ ] A última célula escreve a tabela do mapa, as contagens e a sensibilidade em `results/mapa_estrutura/` e imprime o veredito por estrato
- [ ] Gate check passes: `$PY scripts/check_notebook.py notebooks/item22_mapa_estrutura_colab.ipynb` e o Build gate completo
- [ ] Test count: 22+ tests still pass (no silent deletions)

**Tests**: none
**Gate**: build

**Commit**: `feat(mapa): add Colab notebook that runs the structure map`

---

## Phase Execution Map

```
Phase 1:  T1 → T2 → T3 → T4 → T5 → T6 → T7
```

Execução estritamente sequencial, inline (7 tasks cabem numa leva).

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: assign_strata | 1 função | ✅ Granular |
| T2: sem_serie + contagens | 1 parâmetro e 1 função, mesmo arquivo | ✅ Granular |
| T3: compare_cell | 1 função | ✅ Granular |
| T4: label_cell + holm_family | 2 funções coesas, mesmo arquivo | ⚠️ OK, coesas |
| T5: build_map + structure_pays | 2 funções coesas, mesmo arquivo | ⚠️ OK, coesas |
| T6: três sensibilidades | 3 funções pequenas, mesmo arquivo | ⚠️ OK, coesas |
| T7: notebook + script de gate | 1 notebook | ✅ Granular |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | início da Phase 1 | ✅ Match |
| T2 | T1 | T1 → T2 | ✅ Match |
| T3 | T2 | T2 → T3 | ✅ Match |
| T4 | T3 | T3 → T4 | ✅ Match |
| T5 | T4 | T4 → T5 | ✅ Match |
| T6 | T5 | T5 → T6 | ✅ Match |
| T7 | T6 | T6 → T7 | ✅ Match |

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1: assign_strata | funções puras de avaliação | unit | unit | ✅ OK |
| T2: sem_serie e contagens | funções puras de avaliação | unit | unit | ✅ OK |
| T3: compare_cell | funções puras de avaliação | unit | unit | ✅ OK |
| T4: label_cell e Holm | funções puras de avaliação | unit | unit | ✅ OK |
| T5: build_map | funções puras de avaliação | unit | unit | ✅ OK |
| T6: sensibilidades | funções puras de avaliação | unit | unit | ✅ OK |
| T7: notebook | notebook Colab | none | none | ✅ OK |
