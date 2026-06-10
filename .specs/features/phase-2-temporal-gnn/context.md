# Phase 2 — Context (decisões de gray areas)

Capturado durante o `specify` (2026-05-30). Decisões do pesquisador sobre áreas cinza
que moldam a spec.

## D1 — Objetivo de treino: **Ambos**

Treina e avalia **forecasting 1-passo** (prepara predição genuína da Phase 3 Modo 2)
**e** **fit retroativo da curva** (prepara comparação 1-pra-1 com SIR da Phase 3 Modo 1).
Os dois compartilham o mesmo encoder; diferem no protocolo de dados/avaliação.

**Implicação:** R2 com dois sub-objetivos; saídas de val para ambos (C8); design precisa
definir como o mesmo encoder serve os dois protocolos sem duplicar treino desnecessariamente.

## D2 — Barra de conclusão: **Bater persistência no val**

"Phase 2 concluída" = melhor config do GNN **supera o baseline de persistência ingênua**
`ŷ(t)=y(t-1)` no val MSE, em ambos os regimes (virality e success). A comparação rigorosa
contra SIR (RMSE médio, IC 95%, Mann-Whitney, boxplot Fig. 3) é **deferida à Phase 3**.

**Implicação:** C6/C7 medem GNN vs persistência, não GNN vs SIR. Mantém a Phase 2 focada
em "o modelo aprende sinal temporal" sem antecipar a avaliação científica completa.

## D3 — Hiperparâmetros: **Grid pequeno do ROADMAP**

`W ∈ {4,8,12}`, `hidden ∈ {64,128}`, `layers ∈ {2,3}`, `lr ∈ {1e-3,5e-4}`.
Seleção por val MSE. Sem busca automatizada (Optuna) nem grid ampliado nesta fase.

**Implicação:** R4.2 + C5 exigem o grid completo executado e tabelado. Custo de CPU a
controlar via OQ2 (cache de embeddings por semana) no design.
