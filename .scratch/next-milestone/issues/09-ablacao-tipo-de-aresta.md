# 09 — Ablação por tipo de aresta sobre a GNN final

**What to build:** a variação de erro ao remover cada tipo de relação do grafo, medida em tempo de avaliação sobre o modelo final, nos dois recortes. Responde qual relação carrega o sinal, e cumpre o que o capítulo de cronograma prometeu à banca.

O ticket 04 já terá estabelecido se o instrumento mede alguma coisa; aqui ele é aplicado ao modelo novo e reportado.

**Blocked by:** 06

**Status:** ready-for-agent

- [ ] Delta de erro por tipo de aresta, nos dois recortes e nos dois regimes de chart
- [ ] Resultado agregado entre as seeds do modelo final
- [ ] Deltas exatamente zero, se ocorrerem, são reportados como falha de sensibilidade e não como ausência de efeito
