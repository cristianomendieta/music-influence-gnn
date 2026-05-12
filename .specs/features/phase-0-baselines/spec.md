# Phase 0 — Reprodução dos baselines (SIR + Wave-based)

**Status:** in_progress
**Janela:** semanas 1–2 (2026-05-02 → 2026-05-16)
**Bloqueia:** Phases 1–4. Sem reprodução, qualquer comparação posterior é contaminada.

## Goal

Reproduzir fielmente os baselines de Oliveira et al. **BraSNAM 2025** (SIR clássico)
e **ASONAM 2025** (Contagious Rhythms / wave-based) sobre o subset de 1.179 músicas
da interseção viral∩hit, atingindo RMSE e p-value compatíveis com os números
reportados nos papers (dentro de tolerância pré-acordada).

## Out of scope

- Construção do grafo heterogêneo (Phase 1).
- Treino de qualquer GNN (Phase 2).
- Predição genuína em horizonte k > 0 (Modo 2 fica para Phase 3).
- Análise interpretativa dos baselines além do mínimo necessário para validar
  a reprodução.

## Requirements

### R0 — Pipeline de pré-processamento

- **R0.1** O pré-processamento deve ser **idêntico** ao paper original:
  rank score → moving average 7 dias → min-max para `[0, 0.5]` → floor `0.001`.
- **R0.2** Implementação deve produzir, para cada música no subset, **duas séries
  temporais** independentes — uma derivada do Top 200 (success) e outra do
  Viral 50 (virality), ambas no domínio diário entre 2017-01-01 e 2021-12-31.
- **R0.3** Saída do módulo de pré-processamento deve ser persistida em disco
  (`data/processed/timeseries.parquet` ou `.npz`) para reuso por todas as fases
  seguintes sem reprocessar.

### R1 — Subset de análise

- **R1.1** Subset de trabalho é a interseção `viral ∩ hit` = 1.179 músicas
  (declarado em PROJECT.md como limitação aceita).
- **R1.2** Cada música nesse subset deve ter (a) série virality, (b) série success,
  (c) features acústicas do MGD+ (validar 100% de cobertura).
- **R1.3** A composição do subset deve ser persistida (lista de `song_id`) para
  garantir reprodutibilidade entre fases.

### R2 — Baseline SIR clássico

- **R2.1** Modelo SIR ajustado **por música** via `scipy.integrate.odeint` para
  resolver a EDO + `scipy.optimize.curve_fit` para estimar β e γ.
- **R2.2** Initial guess: β = γ = 0,5. Bounds: `[0, ∞)`.
- **R2.3** Ajustar separadamente em cada uma das duas séries (virality e success).
- **R2.4** Reportar, para cada música, β, γ, R₀ = β/γ, RMSE do fit e flag de
  convergência do otimizador.

### R3 — Baseline Wave-based (Contagious Rhythms)

- **R3.1** Modelo é uma **soma de M ondas SIR independentes** com tempos de início
  distintos, conforme Oliveira ASONAM 2025.
- **R3.2** O número de ondas M é hiperparâmetro **por música**, escolhido por
  critério de seleção (BIC ou AIC; decidir em design.md).
- **R3.3** Reportar, para cada música: M selecionado, parâmetros das ondas,
  RMSE do fit, flag de convergência.
- **R3.4** Caso a reprodução fiel se mostre inviável dentro do prazo, fallback
  é uma **mistura de Gaussianas** com mesmo critério de seleção de M
  (decisão deve ser registrada em STATE.md antes de implementar o fallback).

### R4 — Critérios numéricos de aceitação

A fase **só é considerada concluída** se os números abaixo forem atingidos no
subset de 1.179 músicas:

| Métrica                       | Alvo (paper original) | Tolerância |
| ----------------------------- | --------------------- | ---------- |
| SIR · RMSE médio virality     | ≈ 0,028               | ± 10%      |
| SIR · RMSE médio success      | ≈ 0,052               | ± 10%      |
| Mann-Whitney p-value (success vs virality, RMSE) | ordem de 1e-60 | mesma ordem |

Para o wave-based, o paper ASONAM relata melhora vs SIR clássico em casos com
re-emergência. Critério mínimo: **RMSE médio do wave-based ≤ RMSE do SIR** no
subset, com p-value Mann-Whitney pareado < 0,01.

### R5 — Reprodutibilidade

- **R5.1** Todos os números reportados devem ser regeráveis com um único comando
  (`python scripts/run_phase0.py` ou equivalente).
- **R5.2** Seeds explícitas onde houver não-determinismo (não esperado nesta fase,
  mas registrar).
- **R5.3** Ambiente de execução documentado (`pyproject.toml` + versão exata
  de Python anotada em STATE.md).

### R6 — Saídas para a Phase 3

- **R6.1** Os parâmetros ajustados (β, γ por música para SIR; M e ondas para
  wave-based) devem ser persistidos em formato consultável
  (`results/phase0/sir_params.parquet`, `results/phase0/wave_params.parquet`).
- **R6.2** RMSE por música também persistido para os boxplots da Phase 3
  (replicar Fig. 3 do paper).

## Acceptance test

A fase passa se:

1. `python scripts/run_phase0.py` roda end-to-end sem erros.
2. Os três alvos da tabela R4 são atingidos dentro da tolerância.
3. Wave-based ≤ SIR em RMSE médio com p < 0,01.
4. Subset de 1.179 músicas validado (todos os IDs presentes nos artefatos).
5. PR único e atômico, com mensagem referenciando spec/design/tasks.

## Open questions (resolved during design)

- *(serão movidos para `context.md` se discuss for triggered durante design)*
- AIC vs BIC para seleção de M no wave-based?
- `curve_fit` é suficiente para ajustar wave-based ou precisamos de
  `differential_evolution` por causa de mínimos locais?
- Escala temporal do fit: dia ou semana? (paper original parece usar dia.)

## Traceability

- Phase 0 deste spec ↔ Seção 4 do PLANO.md.
- Critérios R4 ↔ resultados reportados em Oliveira BraSNAM 2025.
- Wave-based R3 ↔ Oliveira ASONAM 2025.
