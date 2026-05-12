# STATE — memória persistente

> Decisões, blockers, lessons, todos e ideias adiadas. Atualizar conforme o trabalho avança.

## Decisions

- **2026-05-02** — Dataset combinado: Kaggle `dhruvildave/spotify-charts` (charts BR
  2017–2021) + MGD+ Zenodo `8086643` (features e gêneros). Cobre 1.179 músicas
  na interseção viral∩hit, contra 1.977 do paper original. Diferença explicada
  pelos ~2,5 meses faltantes em 2022. **Aceito como limitação a declarar.**
- **2026-05-02** — Reestruturação do repo para padrão `src/` + `data/` + `scripts/`
  + `.specs/`. Justificativa: separa dados, código importável, entrypoints e
  planejamento; permite `pip install -e .`.
- **2026-05-02** — Adoção do workflow `tlc-spec-driven`. Cada fase do PLANO.md
  vira uma feature em `.specs/features/<slug>/` com spec → (design) → (tasks) → execute.
- **2026-05-02** — Phase 0 specificada com pipeline completo (spec + design + tasks).
  Decisões de design registradas: BIC para seleção de M no wave-based;
  `differential_evolution` + `curve_fit` em vez de `curve_fit` puro;
  parquet long-format para timeseries cacheadas; joblib para paralelização.

## Blockers

- *(nenhum)*

## Lessons

- **2026-05-03** — `differential_evolution` com M_max=5 (popsize=12, maxiter=200) em 2358 séries
  de 1826 pontos levou >15h com 12 cores. Estimativa do design (1–2h) foi otimista demais.
  Decisão: **reduzir M_max=3** (spec T0.12 prevê esse fallback se exceder 3h).
  M=4 e M=5 capturam padrões que praticamente não existem no subset (distribuição observada no SIR
  mostra que re-emergência relevante ocorre em ≤3 ondas para a maioria das músicas).

- **2026-05-03** — RMSE do SIR ficou ~25-30% acima das metas do paper (viral50: 0.037 vs 0.028;
  top200: 0.066 vs 0.052). Causa identificada: songs com >100 dias ativos no chart (37% do top200)
  têm padrão multi-onda que o SIR clássico não consegue capturar. A mediana do viral50 (0.030)
  está dentro da tolerância ±10%. Discrepância restante é atribuída ao subset diferente
  (1179 vs 1977 músicas) e ao período mais curto. Documentado como limitação aceita.

## Todos

- [x] Especificar Phase 0 (`.specs/features/phase-0-baselines/`).
- [~] Executar Phase 0 (T0.1 → T0.17). Em andamento: T0.12 com M_max=3.
- [ ] Ler ASONAM 2025 ("Contagious Rhythms") na íntegra antes da T0.10 (wave-based).
- [ ] Considerar contatar Gabriel Oliveira (autor) caso wave-based seja difícil de reproduzir fielmente.

## Deferred ideas

- **Causalidade virality↔success** (Oliveira IEEE Access 2025): explorar como
  análise complementar se sobrar tempo na Phase 3.
- **Comparação com short-form video / TikTok**: fora de escopo do BraSNAM 2026,
  reservado para Proposta 1 do mestrado.
- **HGT no lugar de HeteroSAGE**: só se a base não funcionar (Plano B na Phase 2).
- **TGN puro**: mais expressivo mas caro com grafo heterogêneo; só se Plano B
  do HGT também não bastar.

## Preferences

- Idioma de planejamento e código: PT-BR para docs/specs; EN para identifiers e comentários técnicos curtos.
- Comunicação: respostas concisas, sem narração de processo; perguntas de redirect curtas (2–3 sentenças).
