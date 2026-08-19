# 15 — Arquitetura com atenção (extensão)

**What to build:** um codificador heterogêneo baseado em atenção (HGT, ou atenção por relação) no lugar do HeteroGraphSAGE, mantendo fixos a cabeça temporal, o protocolo de splits e a prevenção de vazamento, para comparação justa.

Cumpre duas funções: verificar se a vantagem preditiva observada é robusta à escolha de arquitetura, e fornecer pesos de atenção por tipo de aresta como leitura interpretativa adicional. Está previsto no cronograma aprovado para outubro e novembro; é extensão, não pré-requisito do argumento central.

**Blocked by:** 10

**Status:** ready-for-agent

- [ ] Codificador com atenção substituindo o HeteroGraphSAGE, resto do protocolo intacto
- [ ] Treinado na mesma matriz e comparável célula a célula com a proposta
- [ ] Pesos de atenção por tipo de aresta extraídos e reportados
- [ ] Conclusão explícita sobre a robustez da vantagem à escolha de arquitetura
