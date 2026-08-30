# Recorte on-chart como leitura principal dos resultados

**Status:** aceito (2026-08-18)

No período de teste, cerca de **95%** dos alvos do Modo 2 são semanas em que a
música não está no chart, preenchidas com o piso. O erro médio, portanto, mede
majoritariamente a capacidade de acertar ausência, não de prever popularidade.
Isso torna o RMSE absoluto pequeno por construção, favorece a persistência (repetir
ausência é trivialmente correto) e comprime as diferenças entre modelos.

O Modo 2 passa a ser reportado em duas leituras: universo completo e recorte
on-chart, com o **recorte on-chart como leitura principal**. O recorte já existia
implementado, mas era aplicado só ao Modo 1.

Há um motivo além da apresentação. A ablação por tipo de aresta rodada em julho de
2026 devolveu variação de erro **exatamente zero** para todos os cinco tipos de
aresta e para todos os grupos de features. A predição é `clamp(y_prev + Δ, 0, 0,5)`,
e o grafo entra apenas por `Δ`; nas semanas de piso, `y_prev` é zero e qualquer `Δ`
negativo é anulado pelo `clamp`. A suspeita é que o instrumento de ablação satura
justamente por causa da dominância das semanas fora do chart. Se confirmado, o
recorte on-chart não é escolha de apresentação: é o que devolve sensibilidade à
medição.

**Alternativa rejeitada:** avaliar apenas on-chart. Descartaria entrada e saída do
chart, que são metade do fenômeno de difusão que o trabalho se propõe a modelar.

**Confirmado em 2026-08-30.** O diagnóstico do item 04
([docs/diagnostico-ablacao.md](../diagnostico-ablacao.md)) mediu a saturação: o `clamp`
está ativo em 82,8% das amostras na leitura completa e em 5,3% no recorte on-chart, e
anula 76,9% contra 3,6% da correção estrutural. O efeito da ablação de `performs` é 26
vezes maior no recorte on-chart. A suspeita registrada acima se verificou, com a
ressalva de que o zero exato de julho tinha outra causa somada: o harness da ablação
entregava um `Δ` constante.
