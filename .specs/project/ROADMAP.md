# ROADMAP — da qualificação à defesa

Janela operacional: **2026-08 → 2027-01**. Qualificação aprovada em ago/2026.
O cronograma abaixo é o que consta no capítulo `6-cronograma` do documento de
qualificação, refinado pelas decisões de 2026-08-18 (ver `docs/adr/`).

Resumo de uma página: [`ROADMAP.md`](../../ROADMAP.md).
Glossário dos termos usados aqui: [`CONTEXT.md`](../../CONTEXT.md).

## Concluído (marco anterior)

| Fase | Entrega | Data |
|---|---|---|
| Phase 0 | Reprodução do baseline SIR (5/5 critérios) | 2026-05-12 |
| Phase 1 | Grafo heterogêneo música–artista–gênero, 6.526 / 1.701 / 530 nós | 2026-05-17 |
| Phase 2 | MusicDiffusionGNN treinado, grid de 24 configs pós-correção de vazamento | 2026-06-28 |
| Phase 3 | Avaliação dupla Modo 1 / Modo 2 contra SIR e persistência | 2026-07-07 |

Detalhe e números em [`STATE.md`](./STATE.md). Artefatos canônicos: `results/phase0/`,
`results/phase1/`, `results/phase2_experimentos_v2/`, `results/phase3/`.

## Marco atual: a estrutura relacional agrega?

A tese continua sendo o modelo proposto, mas passa a ser testável: a rodada abaixo
pode falseá-la, e o que a dissertação passa a dizer em cada desfecho já está
registrado em [ADR-0001](../../docs/adr/0001-precompromisso-de-falseamento.md).

### Phase 4 — Diagnóstico do instrumento de ablação (ago) — CONCLUÍDA 2026-08-30

**Desfecho A**, com uma hipótese C confirmada por cima. Relatório e números em
[`docs/diagnostico-ablacao.md`](../../docs/diagnostico-ablacao.md); instrumento em
`notebooks/item04_diagnostico_saturacao_colab.ipynb` (Colab T4, checkpoint
`W12_h128_l3_lr5e-04`, 98.186 amostras, regime `current`).

- O zero exato de 2026-07-07 era **instrumento quebrado**: o harness encodava a semana
  alvo, que `predict` nunca lê. 0% das posições da janela chegavam ao GRU e `Δ` era a
  constante 0,0032 para todas as amostras. Corrigido em `interpretability.py` (`2c645c6`).
- O `clamp` satura de fato, e sustenta o desfecho A: anula **76,9%** da correção
  estrutural na leitura completa contra **3,6%** no recorte on-chart (clamp ativo em
  82,8% contra 5,3% das amostras). Confirma [ADR-0004](../../docs/adr/0004-recorte-on-chart-como-leitura-principal.md).
- Ablação refeita: `cotrajectory` responde por todo o sinal (`delta_rmse` on-chart
  **+0,0933**, RMSE 0,1005 → 0,1938). Os outros quatro tipos têm efeito **adverso**
  (`delta_rmse` negativo: removê-los reduz o erro).

**Consequências para as fases seguintes:**

1. `results/phase3/interpretability.parquet` mede uma constante. Não é resultado, é
   artefato; não pode ser citado, e a permutação por grupo de features precisa ser
   refeita junto com a Phase 6.
2. ~~**Canal de gênero inerte** (bloqueia a Phase 5)~~ — resolvido em 2026-08-31 pela
   Phase 5: a causa era a média de uma tabela i.i.d. centrada em zero, não a falta de
   caminho. Com os atributos do ADR-0003 e o re-treino, o canal conduz (1,9e−04 contra
   3e−08). A ablação de `cooccurs` precisa ser refeita sobre o checkpoint novo.
3. **Descasamento de densidade treino/avaliação** (afeta a Phase 6): treino com
   `max_cotraj_edges = 30_000` por snapshot, avaliação com o grafo completo (480k–664k
   arestas). Religar a cotrajetória ao acaso **melhora** o erro on-chart (−0,0028), o
   que é adverso à tese mas confundido por esse descasamento. O protocolo da escada
   precisa fixar o mesmo orçamento de arestas nos dois lados, ou reportar as duas
   leituras.

### Phase 5 — Nova representação de gênero e reconstrução do grafo (ago–set) — CONCLUÍDA 2026-08-31

Gênero passou a ser descrito por quatro atributos com fórmula derivados da rede gênero↔gênero,
restritos aos anos de treino do regime ([ADR-0003](../../docs/adr/0003-atributos-de-genero-derivados.md));
o `x_genre` aleatório saiu, o grafo foi reconstruído por regime e C1–C9 estão verdes nos dois.
Relatórios: [`docs/genero-estrutural-retreino.md`](../../docs/genero-estrutural-retreino.md) e
[`docs/sonda-canal-genero.md`](../../docs/sonda-canal-genero.md).

- **A troca não custou desempenho.** `val_mse` 0,000754 contra 0,000749 do grafo antigo (+0,7%),
  na mesma config `W12_h128_l3_lr5e-04` e com as mesmas features de música e artista — dentro da
  dispersão 0,000749–0,000764 das 24 configs da grid v2. Bate a persistência nos quatro pares
  split×chart. Uma seed; as outras duas ficam no item 06.
- **O canal de gênero deixou de ser inerte** (fecha o achado 2 da Phase 4 e o item 21). Com pesos
  treinados, esvaziar `cooccurs` move o embedding de `music` em 1,9e−04 contra 3e−08 do modelo
  antigo. A causa antiga era dupla: `SAGEConv` promedia os vizinhos e a média da tabela aleatória
  i.i.d. se cancelava, e o treino não preservava o resto.
- **Conduzir não é ser útil.** O sinal cai de 24,8% da magnitude do embedding em `genre` para
  0,5% em `music`, e o `val_mse` não se mexeu. A utilidade do gênero é o que a Phase 6 mede.

**Consequência para a Phase 6:** a ablação por tipo de aresta precisa ser refeita sobre o
checkpoint novo. O `delta_rmse` zero de `cooccurs` no item 04 foi medido num canal morto; com o
canal vivo, um zero persistente vira afirmação sobre utilidade, não sobre propagação.

### Phase 6 — Escada de comparação (set–out)

Cinco modelos sob o mesmo protocolo, mesmas features e mesmo eixo semanal:

| Modelo | O que isola |
|---|---|
| Persistência | piso trivial do problema |
| SIR (baseline populacional) | o estado da arte replicado |
| Baseline neural sem grafo | capacidade neural sem estrutura |
| GNN sobre grafo embaralhado | estrutura destruída, resto constante ([ADR-0002](../../docs/adr/0002-grafo-embaralhado-como-controle.md)) |
| MusicDiffusionGNN | a proposta |

Mais ablação por tipo de aresta sobre a GNN completa, em tempo de avaliação, sem
re-treino, com o harness corrigido e a permutação por grupo de features refeita. Inclui
obrigatoriamente `cooccurs` (o zero do item 04 saiu de um canal morto) e a sonda de
`has_genre`/`rev_has_genre`, herdada do item 21.

**1 dos 18 treinos já está feito:** `current`/seed 42 sobre o grafo do ADR-0003, entregue
pela Phase 5 (`val_mse` 0,000754). Faltam `current` seeds 43/44 e as três de `pre_pandemia`
para a proposta, mais os dois outros modelos neurais.

**Matriz:** 3 modelos neurais × 2 splits × 3 seeds = **18 treinos**, mais SIR e
persistência refeitos nos dois splits. Avaliação em 2 recortes (completo e on-chart,
[ADR-0004](../../docs/adr/0004-recorte-on-chart-como-leitura-principal.md)) × 3
horizontes × 2 regimes. Segundo split inteiramente pré-pandemia
([ADR-0005](../../docs/adr/0005-segundo-split-pre-pandemia.md)).

**Estatística:** média e desvio entre seeds, IC bootstrap sobre a **diferença** de
RMSE entre modelos, Wilcoxon reagregando os erros **por música** antes do teste (o
código atual pareia por origem de previsão, o que infla a significância), correção
de Holm entre as células.

### Phase 7 — Arquitetura alternativa com atenção (out–nov)

HGT ou GAT por relação no lugar do HeteroGraphSAGE, cabeça temporal e protocolo
fixos. Verifica se a vantagem é robusta à arquitetura e fornece pesos de atenção
por tipo de aresta. Previsto no cronograma aprovado; é extensão, não pré-requisito.

### Phase 8 — Redação e defesa (dez–jan)

Consolidação da comparação única, discussão, conclusão, defesa e submissão.

## Pendências de texto (em paralelo, não dependem de números novos)

Do `comentarios.txt` da banca:

- Glossário aplicado ao resumo e à metodologia (termos mal definidos). Base: `CONTEXT.md`.
- Intro: explicar o dataset e as limitações que ele impõe à proposta.
- Metodologia: corrigir a caracterização do SIR (a literatura da área usa modelos
  bem mais complexos que o SIR básico replicado aqui), com referências.
- Metodologia: fluxograma do processamento dos dados, deixando claro qual modelo usa quais dados.
- Metodologia: EDA que sustente a imputação por mediana dos atributos acústicos.
- Seção 4.7: separar o que é proposta do que é fundamento; incluir a fórmula da
  co-trajetória e das novas features de gênero.
- Resultados: citar a referência do protocolo de validação automatizada, se houver.
- `PLANO.md`: rebaixar a promessa de atenção por tipo de aresta, hoje escrita como
  entrega e na verdade prevista para a Phase 7.

Dependem dos números novos, ficam para depois: resumo, objetivos da introdução,
capítulo de resultados e discussão.

## Riscos

| Risco | Mitigação |
|---|---|
| ~~O grafo não influencia a predição (desfecho B da Phase 4)~~ | Descartado em 2026-08-30: o grafo altera a predição. Risco residual reformulado abaixo |
| Só a cotrajetória carrega sinal, e a topologia dela não importa (prévia do religamento aleatório) | ADR-0001 já define o que a tese passa a dizer; o veredito é a Phase 6 com treino sobre grafo embaralhado |
| Comparações de topologia confundidas pelo subsample de cotrajetória no treino | Fixar o mesmo orçamento de arestas em treino e avaliação na Phase 6 |
| 18 treinos dependem do Colab, que desconecta | Retomada por config já implementada; checkpoints no Drive |
| ~~Novas features de gênero piorarem o resultado~~ | Descartado em 2026-08-31: +0,7% de `val_mse` numa seed, dentro da dispersão da grid. O risco vivo agora é outro: o gênero conduzir e mesmo assim não contribuir para o erro |
| Split pré-pandemia inverter a ordenação dos modelos | É exatamente o que a checagem existe para detectar |
