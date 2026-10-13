# Phase 6b — Mapa de onde a estrutura paga Specification

**Status:** aprovada em 2026-10-13 (parâmetros confirmados pelo autor). Entrega executável em notebook para o Colab.
**Depende de:** `results/comparacao_final/predicoes_cinco_modelos.parquet` (item 10) e `data/processed/timeseries.parquet`.
**Origem:** [ADR-0001](../../../docs/adr/0001-precompromisso-de-falseamento.md), caso 3: "o trabalho passa a entregar o mapa de **onde** a estrutura paga e reporta equivalência no resto".

## Problem Statement

A escada de comparação deu o caso 3 nos dois splits: a proposta não supera o baseline sem grafo nem o grafo embaralhado na média das células. A média esconde heterogeneidade: a estrutura pode pagar em subpopulações específicas (músicas sem histórico, entradas no chart) e ser irrelevante no resto. O ADR-0001 compromete a dissertação a entregar esse mapa. Sem ele, o resultado nulo é só uma média.

É análise sobre predições que **já existem**: nenhum treino, nenhuma GPU.

## Goals

- [ ] Cada origem de previsão do Modo 2 pertence a exatamente um estrato, definido só por informação anterior à origem.
- [ ] Para cada estrato × horizonte × split, o veredito proposta contra sem_grafo e contra embaralhado (vence, perde, equivalente, inconclusivo), com IC e `p` Holm.
- [ ] Uma tabela final "onde a estrutura paga" que a dissertação pode citar, com a hipótese escrita antes do número.

## Out of Scope

| Feature | Reason |
|---|---|
| Re-treino ou novos checkpoints | O mapa lê predições existentes |
| Alterar o veredito do ADR-0001 | O mapa é complementar: o caso 3 já está decidido pela regra original |
| Estratos definidos por informação futura à origem (por exemplo, duração total do hit) | Não são acionáveis na hora da previsão e condicionam no desfecho |
| Atenção, HGT e qualquer arquitetura nova | É a Phase 7 |
| Mapa para SIR e persistência | A pergunta é sobre estrutura; o contraste é grafo contra sem-grafo |

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
|---|---|---|---|
| Hipótese a priori (escrita antes de ver qualquer número estratificado) | A estrutura paga em **estreia** e **reentrada** (sem histórico próprio, a vizinhança é a única informação); não paga em **maduro mono-onda** (a persistência já resume tudo) | Mecanismo: a GNN é ancorada na persistência e só aprende a correção; onde a persistência é cega (origem no piso), só o grafo traz informação | y |
| Unidade de análise | Estrato × horizonte k, charts agrupados, erro quadrático médio **por música** (uma música em dois charts conta uma vez) | O Modo 2 tem muitas origens por música, que não são independentes (CONTEXT.md). Agrupar charts dá o poder que a célula por chart não tem | y |
| Idade que separa jovem de maduro | H = 8 semanas desde a primeira semana on-chart até a origem | Perto da mediana de 8 a 12 semanas observada nas origens on-chart; valor fixado antes de estratificar | y |
| Separação entre ondas | Duas semanas on-chart pertencem a ondas diferentes se há ≥ 4 semanas seguidas fora do chart entre elas | Evita contar uma queda curta como nova onda | y |
| Tamanho mínimo | Célula com menos de 30 músicas é **insuficiente**: relatada, fora do teste e fora da família do Holm | Abaixo disso o bootstrap por música não tem poder; sem esse corte o mapa vira ruído | y |
| Família da correção | Holm sobre todas as células testadas de cada split, 2 contrastes (sem_grafo e embaralhado) × 5 estratos × 3 horizontes, no máximo 30 | Fixar antes evita o problema da escolha de família que o ADR-0001 deixou aberto | y |
| "Vence" e "perde" | IC bootstrap 95% da diferença de erro quadrático inteiro abaixo (acima) de zero **e** `p` Holm < 0,05 | Mesma regra do item 10, para ser comparável | y |
| Equivalência | IC inteiro dentro de ±5% do erro quadrático médio por música do sem_grafo na célula | 5% de MSE ≈ 2,5% de RMSE, perto das diferenças observadas entre variantes (1% a 5%). É a margem mais arbitrária da spec | y |
| "A estrutura paga" num estrato | Vence **os dois** controles em ≥ 2 de 3 horizontes, **nos dois splits** | O sem_grafo isola capacidade, o embaralhado isola topologia; só ganhar dos dois é atribuível a estrutura relacional; replicar entre splits protege contra o período | y |
| Predições usadas | Média das 3 seeds por origem, como no item 10; sensibilidade por seed individual como análise secundária | Comparabilidade com a tabela do item 10; a dispersão entre seeds não pode sumir do mapa | y |
| Estrato por grau de cotrajetória | Fica como P3, exige o grafo (torch), roda só no Colab | O resto usa só pandas | y |
| Como se entrega | Funções puras em `src/music_diffusion_gnn/evaluation/{strata,mapa}.py` com testes leves, e a execução num notebook Colab `notebooks/item22_mapa_estrutura_colab.ipynb` | Pedido do autor: a execução pesada (bootstrap de 10.000 reamostras em dezenas de células) não roda na máquina local | y |

**Open questions:** none - all resolved or logged above (required before the spec is confirmed). H, a separação de ondas e a margem de equivalência são os três números que mais mudam o resultado.

---

## User Stories

### P1: Estratificação causal das origens ⭐ MVP

**User Story**: Como autor, quero cada origem rotulada com seu estágio de vida no chart, calculado só com o passado, para que o mapa seja acionável na hora da previsão e não condicione no desfecho.

**Why P1**: Sem os estratos não há mapa. E um estrato que usa o futuro invalida a leitura inteira.

**Estratos** (mutuamente exclusivos e exaustivos, avaliados na semana de origem `w`):

| Estrato | Definição na origem |
|---|---|
| `estreia` | nenhuma semana on-chart em `≤ w` para esse (música, chart) |
| `reentrada` | semana `w` fora do chart, com ao menos uma semana on-chart antes |
| `jovem` | `w` on-chart, e `w − primeira_semana_on_chart ≤ 8` |
| `maduro_mono` | `w` on-chart, idade `> 8`, uma única onda até `w` |
| `maduro_multi` | `w` on-chart, idade `> 8`, duas ou mais ondas até `w` |

**Acceptance Criteria**:

1. The system SHALL atribuir a cada origem de `predicoes_cinco_modelos.parquet` exatamente um dos cinco estratos.  <!-- ubiquitous -->
2. The system SHALL calcular o estrato de uma origem usando apenas semanas `≤ w` do `timeseries.parquet`.  <!-- ubiquitous -->
3. WHEN a série de um (música, chart) é truncada em `w` THEN the system SHALL devolver o mesmo estrato que devolveria com a série completa.  <!-- event-driven; teste de propriedade, o vazamento falha aqui -->
4. IF uma origem não tem (música, chart) em `timeseries.parquet` THEN the system SHALL registrá-la como estrato `sem_serie` e excluí-la dos testes, contando-a no relatório.  <!-- unwanted-behavior -->
5. WHEN os estratos são tabulados por split e horizonte THEN the system SHALL emitir o número de origens e de músicas por célula e marcar `insuficiente` quando as músicas forem menos de 30.  <!-- event-driven -->

**Independent Test**: `pytest tests/test_strata.py` com séries sintéticas de estágio conhecido; a tabela de contagens reproduz o total de origens do item 10 (141.504 no `current`, 53.982 no `pre_pandemia`, por modelo).

---

### P1: Veredito por estrato ⭐ MVP

**User Story**: Como autor, quero saber, para cada estrato e horizonte, se a proposta vence, perde ou equivale ao sem_grafo e ao embaralhado, para entregar o mapa que o ADR-0001 promete.

**Why P1**: É o entregável do caso 3.

**Acceptance Criteria**:

1. WHEN uma célula (estrato, k, split) tem ≥ 30 músicas THEN the system SHALL calcular, para cada contraste (sem_grafo, embaralhado), a diferença de erro quadrático médio por música, o IC bootstrap 95% (10.000 reamostras, semente fixa) e o `p` do Wilcoxon pareado.  <!-- event-driven -->
2. The system SHALL aplicar a correção de Holm sobre todas as células testadas de cada split, nos dois contrastes juntos.  <!-- ubiquitous -->
3. WHEN o IC e o `p` Holm satisfazem a regra de "vence" (ou "perde") THEN the system SHALL rotular a célula `vence` (ou `perde`); WHEN o IC cabe em ±5% do MSE do sem_grafo THEN the system SHALL rotular `equivalente`; caso contrário `inconclusivo`.  <!-- complex -->
4. WHEN um estrato vence os dois controles em ≥ 2 de 3 horizontes nos dois splits THEN the system SHALL marcá-lo `estrutura paga`.  <!-- event-driven -->
5. The system SHALL escrever a hipótese a priori, a lista de estratos e os limiares no cabeçalho do relatório antes de qualquer tabela de resultado.  <!-- ubiquitous -->
6. The system SHALL emitir os resultados em `results/mapa_estrutura/*.parquet` e um relatório `docs/mapa-onde-a-estrutura-paga.md`.  <!-- ubiquitous -->

**Independent Test**: rodar o notebook localmente com `results/comparacao_final/`; a tabela tem uma linha por (split, estrato, k, contraste) e o rótulo de cada linha é recomputável a partir de IC e `p` Holm.

---

### P2: Sensibilidade e detalhamento

**User Story**: Como autor, quero saber se o mapa resiste à seed e ao chart, para não carregar à dissertação um padrão que é um artefato da média de seeds ou de um chart só.

**Why P2**: Os testes do item 10 rodam sobre a média de 3 seeds, que esconde a dispersão. O mapa é mais frágil que a média global.

**Acceptance Criteria**:

1. WHEN o mapa principal está pronto THEN the system SHALL repetir o veredito em cada seed isolada e reportar em quantas das 3 o rótulo se repete.  <!-- event-driven -->
2. WHEN o mapa principal está pronto THEN the system SHALL reportar a diferença de RMSE por chart dentro de cada estrato, descritiva e sem teste.  <!-- event-driven -->
3. WHERE a margem de equivalência é variada para 2,5% e 10% THE system SHALL reportar quais rótulos mudam.  <!-- optional-feature -->

**Independent Test**: a tabela de sensibilidade tem as três seeds e as três margens, e os rótulos do mapa principal aparecem nela.

---

### P3: Estrato por grau de cotrajetória

**User Story**: Como autor, quero cruzar o mapa com o grau da música na cotrajetória, para testar a hipótese da Phase 7 de que a média simples dilui a vizinhança onde o grau é alto.

**Why P3**: Alimenta o ADR-0006, mas exige o grafo (torch) e o snapshot da semana de origem.

**Acceptance Criteria**:

1. WHERE o grafo `hetero_full_{regime}.pt` está disponível THE system SHALL rotular cada origem com o tercil de grau de entrada da música na cotrajetória no snapshot `mask_until(g, w)`, com tercis definidos nos anos de treino.  <!-- optional-feature -->
2. WHEN o tercil é calculado THEN the system SHALL usar apenas arestas com `first_seen_week ≤ w`.  <!-- event-driven -->

**Independent Test**: o grau de uma música numa semana cresce de forma monótona com `w` num grafo sintético.

---

## Edge Cases

- IF todas as origens de um estrato num split são `insuficiente` THEN the system SHALL registrar o estrato como não testável nesse split e NÃO marcá-lo `estrutura paga`.
- IF o sem_grafo e o embaralhado têm predição idêntica numa origem (célula sem aresta) THEN the system SHALL manter a origem, porque a diferença nula é informação.
- WHEN uma música aparece nos dois charts THEN the system SHALL promediar o erro quadrático dos dois antes do teste, para contá-la uma vez.
- IF a semana de origem é anterior à primeira semana do `timeseries.parquet` THEN the system SHALL classificá-la `estreia`.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
|---|---|---|---|
| MAP-01 | P1: Estratificação causal | T1 | Implementing |
| MAP-02 | P1: Estratificação causal (sem vazamento) | T1 | Verified (T1) |
| MAP-03 | P1: Estratificação causal (contagens e corte de 30) | T2 | Implementing |
| MAP-04 | P1: Veredito por estrato (teste e IC) | T3 | Implementing |
| MAP-05 | P1: Veredito por estrato (família do Holm, rótulos) | T4 | Implementing |
| MAP-06 | P1: Veredito por estrato ("estrutura paga" e relatório) | Specify | Pending |
| MAP-07 | P2: Sensibilidade (seeds, chart, margem) | Specify | Pending |
| MAP-08 | P3: Grau de cotrajetória | Specify | Pending |

**Coverage:** 8 total, 0 mapped to tasks, 8 unmapped ⚠️

---

## Success Criteria

- [ ] Todos os estratos têm contagem por célula e nenhum usa informação posterior à origem (teste de truncamento verde).
- [ ] A tabela do mapa traz, para cada linha, rótulo, IC, `p` Holm e número de músicas.
- [ ] O relatório declara a hipótese a priori antes dos resultados e diz se foi confirmada, refutada ou inconclusiva, sem reescrevê-la.
- [ ] O veredito do ADR-0001 permanece o caso 3; o mapa só o detalha.
