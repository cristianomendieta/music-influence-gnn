# 08 — GNN sobre grafo embaralhado

**What to build:** a mesma proposta, mesma configuração, mesmas features e mesmo protocolo, treinada sobre um grafo com as arestas religadas aleatoriamente preservando distribuição de grau e tipos de aresta. Só a topologia muda.

É o controle que isola estrutura: a diferença entre a GNN real e a embaralhada é, por construção, atribuível à topologia, sem o confundidor de comparar arquiteturas diferentes. Ver `docs/adr/0002-grafo-embaralhado-como-controle.md`.

**Blocked by:** 05, 06

**Status:** ready-for-agent

- [ ] O embaralhamento preserva distribuição de grau e contagem por tipo de aresta, verificado por teste
- [ ] O embaralhamento respeita a restrição temporal das arestas, sem criar vazamento
- [ ] Mesma configuração e mesmo orçamento de treino da proposta
- [ ] Seis treinos concluídos e avaliados na mesma matriz
