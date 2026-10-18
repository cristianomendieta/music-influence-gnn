# Phase 7 — Arquitetura com atenção Specification

**Status:** especificada e aprovada em 2026-10-13, com o [ADR-0006](../../../docs/adr/0006-precompromisso-fase-atencao.md) aceito. Treino e avaliação rodam em notebooks Colab; nada pesado roda na máquina local.
**Depende de:** escada fechada (itens 06 a 10), checkpoints e grafos em `data/processed/graph/hetero_full_{regime}.pt`.
**Substitui:** item 15 (tickets 15, 18 e a parte "convolução que lê `edge_attr`" do 18).

## Problem Statement

A GNN com `HeteroGraphSAGE` não supera o baseline sem grafo nem o grafo embaralhado. Não se sabe se a estrutura relacional não agrega ou se o encoder não sabe usá-la: o `SAGEConv` promedia os vizinhos com peso igual (grau de entrada de até 280 na cotrajetória) e não lê atributo de aresta, então a distinção entre Viral 50 e Top 200 morre na entrada e o vizinho que se acompanha nos dois charts entra com peso dobrado. Um encoder com atenção por relação, lendo atributos de aresta causais, é o teste direto dessa segunda leitura. O resultado, qualquer que seja, entra na dissertação segundo as regras do ADR-0006.

## Goals

- [ ] Um `HeteroAttentionEncoder` (GATv2 por relação) que substitui o `HeteroSpatialEncoder` sem tocar na cabeça temporal, no protocolo ou na prevenção de vazamento.
- [ ] 12 treinos novos (atenção real e atenção embaralhada × 2 splits × 3 seeds) mais o ajuste fixo de 8 configurações.
- [ ] O desfecho O1 a O4 do ADR-0006, por split, com IC e `p` Holm, e a sensibilidade por seed.
- [ ] Pesos de atenção por relação extraídos e descritos, sem uso causal.

## Out of Scope

| Feature | Reason |
|---|---|
| HGT como variante decisiva | Não lê `edge_attr`, não devolve pesos de atenção, devolve `None` para tipos sem mensagem. Robustez secundária (P3), não decide |
| Cabeça temporal sensível ao chart (item 19) | Muda duas coisas ao mesmo tempo. Fica como extensão após o resultado |
| `rev_performs` e outras políticas de direção de aresta | Mexe no grafo, não no encoder |
| `days_together` e `avg_position_distance` brutos como atributo | Calculados sobre o período inteiro, vazam o futuro (ver ADR-0006) |
| Recalcular esses atributos "até a semana `w`" | Pipeline de grafo novo; extensão pós-resultado |
| Retreinar o sem_grafo | Sem arestas o `A` colapsa no mesmo MLP |
| Qualquer variante adicionada depois de ver o teste | Multiplicidade sem correção (ADR-0006) |

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
|---|---|---|---|
| Variante principal | `GATv2Conv` por relação dentro de um `HeteroConv`/laço próprio, `add_self_loops=False`, mais termo raiz por tipo de nó | Lê `edge_attr`, devolve pesos (`return_attention_weights`), e sem arestas colapsa no sem_grafo | y |
| Atributos de aresta | `chart` de origem e `log1p(w − first_seen_week)` apenas | As colunas brutas vazam o futuro. Estas duas são calculáveis no snapshot de `w` | y |
| Relações sem `edge_attr` causal (`performs`, `has_genre`, `rev_has_genre`, `cooccurs`) | Atenção só por conteúdo (`edge_dim=None`) | `cooccurs` tem `weight` vindo da rede de gêneros (`graph/edges.py:301`); a restrição causal dele não foi auditada, então não entra | y |
| Grade de ajuste | 8 configs: heads {2,4} × dropout de atenção {0, 0,2} × lr {5e-4, 1e-3}, `hidden=128`, seleção por `val_mse` em `current`/seed 42 | Fixa antes do teste. Menor que as 24 da proposta, e isso é limitação declarada | y |
| Empate no ajuste | Diferença de `val_mse` ≤ 0,000003 conta como empate e vence a configuração com menos parâmetros | O desvio entre seeds da proposta SAGE no item 06 foi de 0,000002 a 0,000004; o limiar fica fixado antes de ver a grade | y |
| Orçamento de parâmetros | Dentro de ±10% dos 453.121 do SAGE; o número exato é declarado | Comparabilidade; fora disso muda-se o canal por cabeça, não os `heads` | y |
| Controle de topologia | `A-emb` reutiliza `rewire_preserving_degree` e fica com `edge_attr` alinhado à aresta original | Preserva a marginal dos atributos e destrói só o par; `idade` e `chart` continuam coerentes com `first_seen_week` | y |
| Referência sem grafo | O sem_grafo SAGE existente | Mesma classe funcional do `A` sem arestas. O teste mapeia os pesos: `W_raiz,t = Σ_r lin_r` das relações que chegam ao tipo `t`, `b_raiz,t = Σ_r b_lin_l`. Paridade de função, não de otimização | y |
| Orçamento de arestas na avaliação | Duas leituras: grafo completo (principal) e `teto_treino` | Mesma regra do ticket 08 | y |
| Onde roda | Encoder e testes leves em `src/`; ajuste, treinos e avaliação em notebooks Colab (GPU) | Pedido do autor: nada pesado na máquina local, que não tem torch utilizável | y |
| Prazo | Matriz até 2026-11-27 | Deixa dezembro e janeiro para redação; se não fechar, reporta parcial | y |

**Open questions:** none - all resolved or logged above (required before the spec is confirmed). Grade de 8 configs e prazo de 27/nov confirmados pelo autor.

---

## User Stories

### P1: Encoder de atenção causal ⭐ MVP

**User Story**: Como autor, quero um encoder com atenção por relação que lê só atributos de aresta disponíveis no snapshot, para testar a hipótese do ADR-0006 sem vazamento.

**Why P1**: Sem o encoder e sem o teste de vazamento, nenhum treino pode ser interpretado.

**Acceptance Criteria**:

1. The system SHALL oferecer `MusicDiffusionGNN(encoder="gatv2")` mantendo `encoder="sage"` como padrão e o comportamento atual intacto.  <!-- ubiquitous -->
2. WHEN o encoder recebe um snapshot `mask_until(g, w)` THEN the system SHALL usar como atributo de aresta somente `chart` e `log1p(w − first_seen_week)`.  <!-- event-driven -->
3. WHEN as colunas `days_together` e `avg_position_distance` de `edge_attr` são perturbadas THEN the system SHALL devolver embeddings idênticos aos originais.  <!-- event-driven; portão de vazamento -->
4. WHEN o grafo tem todas as arestas removidas (`strip_all_edges`) THEN the system SHALL devolver o mesmo embedding de música que o `HeteroSpatialEncoder` com os mesmos pesos do ramo raiz.  <!-- event-driven; paridade com o sem_grafo -->
5. WHEN `return_attention_weights` é pedido THEN the system SHALL devolver, por relação, pesos de atenção que somam 1 por nó de destino com ao menos uma aresta de entrada.  <!-- event-driven -->
6. The system SHALL reportar o número de parâmetros do encoder e abortar o treino se ele ficar fora de ±10% de 453.121.  <!-- ubiquitous -->
7. IF uma relação não tem aresta no snapshot THEN the system SHALL omitir sua contribuição sem lançar erro e sem devolver `None` para o tipo de nó.  <!-- unwanted-behavior -->

**Independent Test**: `pytest tests/test_phase7_encoder.py` com um grafo sintético pequeno; o teste 3 falha de propósito se o encoder ler a coluna bruta.

---

### P1: Treino e avaliação da matriz ⭐ MVP

**User Story**: Como autor, quero os 12 treinos e o ajuste com o mesmo protocolo da escada, retomáveis, para comparar célula a célula com o item 10.

**Why P1**: É o dado do experimento.

**Acceptance Criteria**:

1. WHEN o ajuste roda THEN the system SHALL treinar as 8 configurações da grade em `current`/seed 42 e registrar `val_mse` e parâmetros de cada uma antes de qualquer avaliação em teste.  <!-- event-driven -->
2. The system SHALL usar a configuração escolhida no ajuste para os 12 treinos, sem alterá-la depois de ver o teste.  <!-- ubiquitous -->
3. WHEN um treino é interrompido THEN the system SHALL retomar do checkpoint existente sem retreinar os que já terminaram.  <!-- event-driven -->
4. The system SHALL gravar, por checkpoint, o regime de split, a seed, a variante (`A` ou `A-emb`), a configuração e a versão do grafo.  <!-- ubiquitous -->
5. WHEN o Modo 2 roda THEN the system SHALL avaliar k ∈ {1, 2, 4}, nos dois charts, nos recortes completo e on-chart, nas duas leituras de orçamento de arestas.  <!-- event-driven -->
6. WHILE o treino roda, the system SHALL fixar `max_cotraj_edges = 30_000` e o mesmo número de épocas (35) da escada.  <!-- state-driven -->

**Independent Test**: o notebook de treino lista 12 checkpoints e a tabela de `val_mse` por (variante, split, seed); o de avaliação reproduz o formato de `predicoes_cinco_modelos.parquet` com as variantes novas.

---

### P1: Veredito do ADR-0006 ⭐ MVP

**User Story**: Como autor, quero o desfecho O1 a O4 por split, com a mesma estatística do item 10, para reescrever a conclusão sem racionalizar.

**Why P1**: É o entregável científico.

**Acceptance Criteria**:

1. WHEN as predições dos sete modelos estão alinhadas THEN the system SHALL calcular H1, H2 e H3 por split e por célula (chart × k) on-chart, com IC bootstrap 95% e `p` do Wilcoxon sobre o erro quadrático médio por música.  <!-- event-driven -->
2. The system SHALL aplicar o Holm sobre as 18 comparações de cada split.  <!-- ubiquitous -->
3. WHEN H1 e H2 estão em maioria (≥ 4 de 6) THEN the system SHALL declarar O1; WHEN só H1 THEN O2; WHEN nenhuma das duas e `A` não perde em maioria THEN O3; WHEN `A` perde para o sem_grafo em maioria THEN O4.  <!-- event-driven -->
4. IF os dois splits levam a desfechos diferentes THEN the system SHALL reportar a discordância como resultado e restringir a afirmação principal ao que se replica.  <!-- unwanted-behavior -->
5. WHEN o veredito é calculado THEN the system SHALL repetir H1 a H3 em cada seed isolada e reportar a concordância com o teste sobre a média.  <!-- event-driven -->
6. The system SHALL reportar o orçamento de ajuste (8 contra 24 configurações) e o número de parâmetros no mesmo bloco da tabela de resultados.  <!-- ubiquitous -->

**Independent Test**: `veredito_adr0006.parquet` tem uma linha por (split, contraste) com o desfecho, e cada rótulo é recomputável a partir de `testes_pareados`.

---

### P2: Pesos de atenção

**User Story**: Como autor, quero descrever como a atenção se distribui na cotrajetória, para ilustrar o que a estrutura aprendeu sem afirmar que isso explica a previsão.

**Why P2**: Cumpre a promessa do cronograma sobre interpretabilidade, mas não decide nada.

**Acceptance Criteria**:

1. WHEN os pesos são extraídos on-chart THEN the system SHALL calcular a entropia normalizada da atenção por nó de destino e seu cruzamento com o grau de entrada, para `A` e `A-emb`.  <!-- event-driven -->
2. The system SHALL rotular toda figura de atenção como descritiva e remeter à ablação por tipo de aresta como medida causal.  <!-- ubiquitous -->
3. WHERE a Phase 6b P3 (grau de cotrajetória) está disponível THE system SHALL cruzar o ganho do `A` sobre o SAGE com o tercil de grau.  <!-- optional-feature -->

**Independent Test**: figura de entropia por grau para `A` e `A-emb`, com legenda descritiva.

---

### P3: Ablação por tipo de aresta no `A`

**User Story**: Como autor, quero repetir a ablação por tipo de aresta e a permutação por grupo de features no `A`, para comparar com a do SAGE.

**Why P3**: Depende do desfecho; se for O3, a ablação importa menos.

**Acceptance Criteria**:

1. WHERE o desfecho é O1 ou O2 THE system SHALL repetir a ablação com o harness corrigido do item 04 e reportar nenhum delta exatamente zero como falha de sensibilidade, não como ausência de efeito.  <!-- optional-feature -->

**Independent Test**: tabela de deltas por componente, mesmo formato do item 09.

---

## Edge Cases

- IF o ajuste devolve duas configurações com `val_mse` empatado dentro da dispersão da seed THEN the system SHALL escolher a de menor número de parâmetros e registrar o empate.
- IF a atenção não converge (perda NaN) numa seed THEN the system SHALL registrar a falha, repetir com a mesma configuração e outra inicialização uma vez, e reportar as duas tentativas.
- WHEN o grafo tem arestas paralelas (par nos dois charts) THEN the system SHALL tratar cada uma como aresta própria, com seu `chart`, e não somá-las antes da atenção.
- IF a memória da GPU não comporta a avaliação com 664.577 arestas THEN the system SHALL avaliar em lotes de semanas e registrar a mudança, sem alterar o orçamento de arestas.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
|---|---|---|---|
| ATT-01 | P1: Encoder causal (interface, atributos, vazamento, paridade) | Specify | Pending |
| ATT-02 | P1: Encoder causal (pesos de atenção, orçamento, relação vazia) | Specify | Pending |
| ATT-03 | P1: Treino e avaliação (ajuste e matriz) | Specify | Pending |
| ATT-04 | P1: Treino e avaliação (retomada, metadados, protocolo) | Specify | Pending |
| ATT-05 | P1: Veredito do ADR-0006 | Specify | Pending |
| ATT-06 | P2: Pesos de atenção | Specify | Pending |
| ATT-07 | P3: Ablação no `A` | Specify | Pending |

**Coverage:** 7 total, 6 mapped to tasks (ATT-07 não entra: depende do desfecho), 1 aguardando desfecho

---

## Success Criteria

- [ ] O teste de invariância às colunas brutas de `edge_attr` passa antes do primeiro treino.
- [ ] A matriz de 12 treinos está completa, ou a parte faltante é declarada, até 2026-11-27.
- [ ] O desfecho O1 a O4 sai por split de uma regra escrita antes do número, e cada seed isolada é reportada.
- [ ] A dissertação diz qual das duas leituras do nulo (estrutura não agrega, encoder não aproveita) os dados sustentam.
