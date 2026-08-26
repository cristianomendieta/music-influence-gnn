# 20 — Limitações de arquitetura declaradas no texto

**What to build:** o texto passa a declarar as limitações do codificador que hoje só existem como resposta oral. O item 13 fez isso para o dataset; este faz para a arquitetura.

Quatro, todas verificadas no código:

**A ponderação por relação é global, não por caso.** Cada tipo de aresta tem as suas matrizes, mas o `HeteroConv` combina os canais com `aggr="sum"` (`encoder.py:36`): nada olha a vizinhança de uma música específica e redistribui importância entre relações. A adaptatividade por caso existe só no eixo do tempo, nos portões da GRU. A frase que resume, e que vale para o texto: o modelo é **adaptativo no tempo e fixo nas relações**. É a lacuna que o item 15 preencheria.

**A agregação por média descarta o grau.** Uma música com 2 predecessoras e outra com 200 entregam vetores de escala parecida, e nenhum atributo de nó guarda a contagem, embora grau alto de cotrajetória seja sinal de permanência no chart. A contrapartida é real e deve aparecer junto: é a média que torna coerente treinar com o grafo podado em 30.000 arestas e avaliar com as 664.577.

**O protocolo é indutivo no tempo e transdutivo nos nós.** A divisão é por semana e as 6.526 músicas do teste estão todas no grafo desde a construção. A arquitetura permitiria avaliar música inédita, o experimento não faz isso. O texto não pode sugerir que faz.

**O cold start é severo e não é medido.** O `first_seen_week` de `performs` é a primeira semana da própria música em chart (`edges.py:37`), então antes de estrear a música é **nó isolado** no grafo mascarado: sem vizinho para agregar, com âncora zero, o modelo degenera para um MLP sobre os atributos acústicos. E o canal que carrega a hipótese do trabalho, a cotrajetória, exige sete dias acumulados em chart, ou seja, é justamente o que falta no lançamento. Como o caso de uso que motiva o trabalho é detectar música em ascensão, essa limitação precisa estar escrita, não improvisada.

Contexto em `docs/achados-qualificacao.md`, itens C4, C5 e C6.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] As quatro limitações aparecem no texto, com o caminho no código que as sustenta
- [ ] Fica explícito o que cada uma impede o trabalho de afirmar
- [ ] A contrapartida da média (treinar podado, avaliar completo) aparece junto da limitação, não separada
- [ ] Nenhuma delas é apresentada como defeito de implementação; são escolhas com custo declarado
