# 09 — Ablação por tipo de aresta sobre a GNN final

**What to build:** a variação de erro ao remover cada tipo de relação do grafo, medida em tempo de avaliação sobre o modelo final, nos dois recortes. Responde qual relação carrega o sinal, e cumpre o que o capítulo de cronograma prometeu à banca.

O ticket 04 estabeleceu que o instrumento antigo **não** media nada: `_predict_all`
encodava a semana alvo em vez da janela e `Δ` chegava constante ao GRU (corrigido em
`2c645c6`). Aqui o harness corrigido é aplicado ao modelo novo e reportado.

A permutação por grupo de features tem o mesmo defeito de origem e nunca foi medida de
verdade: refazer junto. E os resultados de `results/phase3/interpretability.parquet` são
artefato, não podem ser citados.

**Blocked by:** 06

**Status:** ready-for-agent

- [ ] Delta de erro por tipo de aresta, nos dois recortes e nos dois regimes de chart
- [ ] Resultado agregado entre as seeds do modelo final
- [ ] Deltas exatamente zero, se ocorrerem, são reportados como falha de sensibilidade e não como ausência de efeito
- [ ] Permutação por grupo de features refeita com o harness corrigido
