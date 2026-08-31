# 06 — MusicDiffusionGNN re-treinada na nova matriz

**What to build:** a proposta treinada e avaliada sob o protocolo completo do novo marco: 2 regimes de split × 3 seeds, avaliada nos 2 recortes × 3 horizontes × 2 regimes de chart, com média e desvio entre seeds.

Os resultados anteriores não são comparáveis: mudaram as features de gênero, entrou o segundo split e entrou o recorte on-chart.

**Blocked by:** 02, 03, 05

**Status:** ready-for-agent — **1 dos 6 treinos já está feito**. O item 05 rodou `current`/seed 42
sobre o grafo do ADR-0003 (`val_mse` 0,000754, checkpoint `gnn_current_seed42.pt` no Drive e em
`results/item05_genero_estrutural/`). Faltam `current` seeds 43 e 44 e as três seeds de
`pre_pandemia`. O notebook do item 05 já retoma por seed e por regime.

- [ ] Seis treinos concluídos (2 splits × 3 seeds), com retomada em caso de desconexão
- [ ] Avaliação nos dois recortes e nos três horizontes, para os dois regimes de chart
- [ ] Resultados reportados como média e desvio entre seeds, não como número único
- [ ] Artefatos registram regime de split, seed e versão do grafo que os gerou
