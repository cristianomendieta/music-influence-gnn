# 03 — Estatística correta

**What to build:** a comparação entre modelos passa a usar o teste apropriado à unidade de observação e a reportar incerteza sobre a diferença, não sobre cada modelo isolado.

Três correções. Primeira: o Wilcoxon do Modo 2 hoje pareia erros por **origem de previsão**, não por música, o que gera ~23 mil pares em vez de ~1.900 e infla a significância, já que origens da mesma música não são independentes; os erros passam a ser agregados por música antes do teste. Segunda: o intervalo de confiança bootstrap passa a ser calculado sobre a **diferença** de erro entre dois modelos. Terceira: correção de Holm entre as células comparadas, e agregação de média e desvio entre seeds.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] O teste pareado do Modo 2 opera sobre erro por música, não por origem de previsão
- [ ] Cada comparação entre dois modelos reporta IC bootstrap da diferença
- [ ] Valores-p corrigidos por Holm dentro de cada família de comparações
- [ ] A rotina aceita múltiplas seeds e reporta média e desvio
- [ ] Recomputação sobre os resultados atuais confirma a faixa de vitória da GNN já conhecida (64% a 77%)
