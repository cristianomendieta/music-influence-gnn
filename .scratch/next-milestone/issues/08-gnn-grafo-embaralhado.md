# 08 — GNN sobre grafo embaralhado

**What to build:** a mesma proposta, mesma configuração, mesmas features e mesmo protocolo, treinada sobre um grafo com as arestas religadas aleatoriamente preservando distribuição de grau e tipos de aresta. Só a topologia muda.

É o controle que isola estrutura: a diferença entre a GNN real e a embaralhada é, por construção, atribuível à topologia, sem o confundidor de comparar arquiteturas diferentes. Ver `docs/adr/0002-grafo-embaralhado-como-controle.md`.

**Blocked by:** 05, 06

**Status:** ready-for-agent

- [ ] O embaralhamento preserva distribuição de grau e contagem por tipo de aresta, verificado por teste
- [ ] O embaralhamento respeita a restrição temporal das arestas, sem criar vazamento
- [ ] Mesma configuração e mesmo orçamento de treino da proposta
- [ ] Seis treinos concluídos e avaliados na mesma matriz

---

## Prévia barata (2026-08-30, item 04)

Religar a cotrajetória ao acaso **em tempo de avaliação**, preservando a contagem de
arestas e os nós de origem, **reduz** o RMSE on-chart em 0,00284 (0,10049 → 0,09765) e é a
maior melhora pré-clamp de todas as variantes (−0,00847). Adverso à tese, e é a razão de
este ticket existir.

Não conclui nada, por dois motivos que este ticket precisa resolver no protocolo:

1. sem re-treino, o religamento é perturbação fora da distribuição de treino, não um
   modelo alternativo;
2. o treino roda com `max_cotraj_edges = 30_000` por snapshot e a avaliação usa o grafo
   completo (480k–664k arestas). O modelo nunca viu a topologia real inteira: um
   religamento aleatório pode simplesmente se parecer mais com o que ele viu. **Fixar o
   mesmo orçamento de arestas em treino e avaliação, ou reportar as duas leituras.**

- [ ] Orçamento de arestas de cotrajetória igual em treino e avaliação, ou as duas leituras reportadas
