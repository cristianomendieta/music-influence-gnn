# 06 — MusicDiffusionGNN re-treinada na nova matriz

**What to build:** a proposta treinada e avaliada sob o protocolo completo do novo marco: 2 regimes de split × 3 seeds, avaliada nos 2 recortes × 3 horizontes × 2 regimes de chart, com média e desvio entre seeds.

Os resultados anteriores não são comparáveis: mudaram as features de gênero, entrou o segundo split e entrou o recorte on-chart.

**Blocked by:** 02, 03, 05

**Status:** ready-for-agent

- [ ] Seis treinos concluídos (2 splits × 3 seeds), com retomada em caso de desconexão
- [ ] Avaliação nos dois recortes e nos três horizontes, para os dois regimes de chart
- [ ] Resultados reportados como média e desvio entre seeds, não como número único
- [ ] Artefatos registram regime de split, seed e versão do grafo que os gerou
