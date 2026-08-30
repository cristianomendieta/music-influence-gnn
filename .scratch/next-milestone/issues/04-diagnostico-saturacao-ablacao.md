# 04 — Diagnóstico da saturação da ablação

**What to build:** um relatório que responde se a ablação por tipo de aresta mede alguma coisa. Hoje ela devolve variação de erro **exatamente zero** para os cinco tipos de aresta e para os três grupos de features, na rodada completa de 1.955 músicas.

A predição é `clamp(y_prev + Δ, 0, 0,5)` e o grafo entra apenas por `Δ`. Nas semanas de piso, `y_prev` é zero e qualquer `Δ` negativo é anulado pelo clamp inferior; com ~95% do conjunto de avaliação nessas semanas, o instrumento pode saturar por construção.

O relatório precisa distinguir dois desfechos. **A:** o instrumento satura no clamp, e o recorte on-chart devolve sensibilidade; segue o plano. **B:** o grafo não influencia a predição em recorte nenhum, e então o ganho da GNN sobre o SIR vem do ancoramento à persistência, o que muda a leitura de toda a avaliação já publicada na qualificação.

Este ticket bloqueia o marco inteiro: nenhum treino novo antes do veredito.

**Blocked by:** 01 (o diagnóstico precisa repetir a ablação sob recorte on-chart)

**Status:** done (2026-08-30)

- [x] Quantifica, no checkpoint atual, quanto da predição vem de `y_prev` e quanto de `Δ`
- [x] Mostra se as predições diferem com e sem grafo **antes** do clamp
- [x] Reporta a fração de amostras em que o clamp está ativo
- [x] Repete a ablação por tipo de aresta sob recorte on-chart
- [x] Conclui explicitamente por desfecho A ou B, com o número que sustenta a conclusão

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

---

## Resultado (2026-08-30) — desfecho A, hipótese C confirmada

Rodado no Colab T4, checkpoint `W12_h128_l3_lr5e-04`, regime `current`, 98.186 amostras
sobre 1.955 músicas (4,6% on-chart). Relatório completo com todas as tabelas em
[`docs/diagnostico-ablacao.md`](../../../docs/diagnostico-ablacao.md).

- **C confirmada.** O harness antigo encontrava 0% das posições da janela no banco e
  devolvia `Δ` constante em 0,00320397 para todas as amostras; a montagem correta dá
  36.314 valores distintos. A ablação de jul/2026 não mediu o grafo.
- **A confirmada.** O `clamp` está ativo em 82,8% das amostras na leitura completa e em
  5,3% no recorte on-chart; anula 76,9% contra 3,6% da correção estrutural. O efeito da
  ablação de `performs` é 26 vezes maior on-chart do que na leitura completa.
- **Ablação refeita:** `cotrajectory` responde por todo o sinal (`delta_rmse` on-chart
  +0,0933, RMSE 0,1005 → 0,1938). Os outros quatro tipos têm `delta_rmse` negativo.

Desdobramentos: item **21** (canal de gênero inerte, bloqueia 05), nota no item **08**
(o religamento aleatório melhora o erro) e no item **09** (harness corrigido, permutação
de features a refazer).
