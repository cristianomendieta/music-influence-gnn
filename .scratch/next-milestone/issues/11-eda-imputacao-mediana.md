# 11 — EDA da imputação por mediana

**What to build:** a análise que sustenta, ou derruba, a decisão atual de imputar os atributos acústicos ausentes pela mediana com indicador binário de ausência. A banca perguntou se houve EDA e se há análise que dê suporte; hoje não há.

Pode mudar código: se a ausência não for aproximadamente aleatória, ou se as músicas sem atributos acústicos tiverem perfil de popularidade distinto, a imputação atual introduz viés e precisa mudar.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Quantifica a ausência por atributo e por ano
- [ ] Compara o perfil de popularidade entre músicas com e sem atributos completos
- [ ] Conclui explicitamente se a imputação por mediana se sustenta
- [ ] Se não se sustentar, a alternativa fica proposta com justificativa
