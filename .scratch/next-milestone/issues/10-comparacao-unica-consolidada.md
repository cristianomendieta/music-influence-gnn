# 10 — Comparação única consolidada

**What to build:** uma tabela só, com os cinco modelos sob as mesmas métricas e o mesmo eixo semanal: persistência, SIR, baseline neural sem grafo, GNN sobre grafo embaralhado e MusicDiffusionGNN. Dois regimes de split, dois recortes, três horizontes, dois regimes de chart, com a estatística do ticket 03 aplicada.

É o insumo do capítulo de resultados e o momento em que o pré-compromisso de `docs/adr/0001` é acionado: a leitura da tabela determina se a tese se sustenta como está, se o ganho vem de agregação e não de topologia, ou se a tese vira condicional.

**Blocked by:** 06, 07, 08, 09

**Status:** ready-for-agent

- [ ] Uma tabela com os cinco modelos, comparáveis célula a célula
- [ ] Cada comparação traz IC da diferença e valor-p corrigido
- [ ] A ordenação dos modelos é reportada separadamente para cada regime de split
- [ ] O desfecho é declarado explicitamente segundo os três casos do pré-compromisso
