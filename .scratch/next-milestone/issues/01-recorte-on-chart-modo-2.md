# 01 — Recorte on-chart no Modo 2

**What to build:** a avaliação preditiva passa a ser reportada em duas leituras: universo completo (todas as semanas do período de teste, incluindo as de piso) e recorte on-chart (só as semanas em que a música está efetivamente no chart). A leitura on-chart é a principal. O recorte já existe aplicado ao Modo 1; aqui ele passa a valer também na previsão causal a k semanas.

Contexto: no período de teste, ~95% dos alvos do Modo 2 são semanas fora do chart, no valor de piso. O erro médio hoje mede majoritariamente acerto de ausência, não previsão de popularidade. Ver `docs/adr/0004-recorte-on-chart-como-leitura-principal.md`.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] O Modo 2 produz, para cada célula (regime × horizonte), o erro nas duas leituras
- [ ] A leitura on-chart aparece como principal nas tabelas geradas
- [ ] Fica registrado quantas origens de previsão sobrevivem ao recorte em cada célula
- [ ] Os números da leitura sem recorte reproduzem os resultados atuais
