# 04 — Diagnóstico da saturação da ablação

**What to build:** um relatório que responde se a ablação por tipo de aresta mede alguma coisa. Hoje ela devolve variação de erro **exatamente zero** para os cinco tipos de aresta e para os três grupos de features, na rodada completa de 1.955 músicas.

A predição é `clamp(y_prev + Δ, 0, 0,5)` e o grafo entra apenas por `Δ`. Nas semanas de piso, `y_prev` é zero e qualquer `Δ` negativo é anulado pelo clamp inferior; com ~95% do conjunto de avaliação nessas semanas, o instrumento pode saturar por construção.

O relatório precisa distinguir dois desfechos. **A:** o instrumento satura no clamp, e o recorte on-chart devolve sensibilidade; segue o plano. **B:** o grafo não influencia a predição em recorte nenhum, e então o ganho da GNN sobre o SIR vem do ancoramento à persistência, o que muda a leitura de toda a avaliação já publicada na qualificação.

Este ticket bloqueia o marco inteiro: nenhum treino novo antes do veredito.

**Blocked by:** 01 (o diagnóstico precisa repetir a ablação sob recorte on-chart)

**Status:** ready-for-agent

- [ ] Quantifica, no checkpoint atual, quanto da predição vem de `y_prev` e quanto de `Δ`
- [ ] Mostra se as predições diferem com e sem grafo **antes** do clamp
- [ ] Reporta a fração de amostras em que o clamp está ativo
- [ ] Repete a ablação por tipo de aresta sob recorte on-chart
- [ ] Conclui explicitamente por desfecho A ou B, com o número que sustenta a conclusão

---

## Andamento (2026-08-26)

Hipótese nova, **C**, levantada na leitura do código antes de qualquer execução:
`_predict_all` (em `evaluation/interpretability.py`) chamava
`model.encode_weeks(g, [s.target_week])`, mas `MusicDiffusionGNN.predict` lê o banco
nas semanas da **janela** `[w-W, …, w-1]` — a semana alvo nunca está nela. Toda posição
da janela caía no ramo `torch.zeros(B, hidden)`, o GRU recebia sequência nula e `Δ`
virava constante: a ablação **não podia** dar outra coisa que zero exato, e o mesmo
vale para a permutação por grupo de features (que também deu zero).

Corrigido em `interpretability.py` (`_predict_all` passa a encodar as semanas da janela,
com cache entre semanas alvo). 12 testes verdes.

Instrumento do veredito: `notebooks/item04_diagnostico_saturacao_colab.ipynb`, autocontido
(reimplementa a predição, não depende da correção estar no `main`), para rodar no Colab GPU.
Mede as três hipóteses e conclui por A, B ou A-parcial com o número que sustenta.
