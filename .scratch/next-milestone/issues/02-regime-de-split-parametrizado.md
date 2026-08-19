# 02 — Regime de split parametrizado (prefactor)

**What to build:** os limites temporais de treino, validação e teste deixam de estar fixos no código e passam a ser configuração nomeada, com dois regimes declarados: o atual (treino até 2020-06-29, teste de 2020-12-28 a 2022-03-13) e o pré-pandemia (treino 2017–2018, validação no primeiro semestre de 2019, teste no segundo semestre de 2019).

Isto é prefactor: sem ele, acrescentar o segundo regime vira uma edição espalhada por cada ticket de treino e avaliação. Ver `docs/adr/0005-segundo-split-pre-pandemia.md`.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Os dois regimes de split são selecionáveis por configuração, sem edição de código
- [ ] Rodar com o regime atual reproduz os números de hoje
- [ ] A verificação de vazamento temporal continua valendo nos dois regimes
- [ ] Os artefatos de resultado registram sob qual regime foram gerados
