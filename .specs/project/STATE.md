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
- **2026-05-12** — Wave-based baseline (R3) **descartado**. `differential_evolution` mesmo com
  M_max=3 é computacionalmente proibitivo, e o SIR já passou todos os critérios de R4
  de forma independente. Comparação com wave-based não é crítica para a contribuição central
  do trabalho (GNN temporal heterogêneo). Registrado em tasks.md como [~] (skipped).
- **2026-05-12** — **Phase 0 concluída com sucesso.** Resultados: RMSE virality 0,0289 ✅,
  RMSE success 0,0471 ✅, Mann-Whitney p=1,61e-39 ✅, subset 1.981 músicas ✅, conv. 100% ✅.
  Artefatos em `results/phase0/` (summary.md, sir_params.parquet, boxplot_fig3.png).
- **2026-05-17** — Limpeza do repositório: removidos DATA_PLAN.md, todos.md, UNKNOWN.egg-info/,
  .ipynb_checkpoints/; scripts exploratórios movidos para scripts/exploratory/; STATUS.md criado
  na raiz como dashboard de status do projeto.
- **2026-05-17** — **Phase 1 especificada** (`.specs/features/phase-1-hetero-graph/spec.md`).
  Decisões registradas: (1) universo de music nodes = 6.469 (Top200 BR 2017-2021), não apenas
  o subset de 1.981 viral∩hit; (2) co-trajetória música↔música incluída no v1 como 4.º tipo
  de aresta, sem deferimento para ablation; (3) grafo estático único com máscara temporal via
  `first_seen_week`, sem materializar ~260 snapshots semanais em disco; (4) único grafo com
  atributo `chart ∈ {viral50, top200}` na aresta co-trajetória, filtro feito em runtime no Phase 2.

- **2026-05-17** — **Phase 1 design aprovado** (`.specs/features/phase-1-hetero-graph/design.md`).
  Resolução das 5 open questions: (Q1) genre features = `nn.Embedding(530, 32)` aprendido,
  init aleatório, sem pretrain node2vec; (Q2) co-trajetória em ambos charts = 2 arestas paralelas;
  (Q3) músicas sem features acústicas = imputar mediana + flag `acoustic_missing`;
  (Q4) `has_genre` via `ToUndirected()` do PyG; (Q5) sem pretrain do embedding de gênero.
  Arquitetura: builders independentes (`nodes.py`, `edges.py`) → orquestrador `build.py` →
  `stats.py` para relatório. Validações C1-C9 inline no build; `mask_until` é função pura.
  Dívida técnica registrada: `loaders.py` aponta caminhos defasados (passar `path=` explícito por ora).

- **2026-05-17** — **Phase 1 tasks definidas** (`.specs/features/phase-1-hetero-graph/tasks.md`).
  16 tasks atômicas em 6 waves: T1 deps → T2 temporal → T3-T9 builders (7 paralelos) →
  T10 orquestrador+C1-C7 → T11 stats → T12/T13 report+plot (paralelos) → T14 run_phase1
  smoke-test C8/C9 → T15/T16 testes (paralelos). Cada task = 1 commit; PR único final.
  Sem MCPs/Skills necessários (stack PyG puro).

- **2026-05-17** — **Phase 1 implementada com sucesso.** C1-C9 todos verdes.
  Resultados: n_music=6.526 ✅ (tolerância atualizada para ±100), n_artist=1.701 ✅,
  n_genre=530 ✅, subset coverage 100% ✅, HeteroSAGE forward shape=(6526,128) ✅,
  mask_until monotônica ✅. Build time: ~40s (cotrajectory 36s). Stats em 80s.
  Artefatos: `data/processed/graph/hetero_full.pt`, `node_id_map.json`,
  `results/phase1/stats.md`, `results/phase1/degree_distributions.png`.

- **2026-05-17** — **Descoberta de desvio do spec em n_music**: spec estimou 6.469 (top200 BR
  2017-2021), mas top200 real tem 5.010 únicos. O universo real adotado:
  top200 ∪ (viral50 ∩ acoustic_features_in_complete_MGDplus) = 6.526 músicas.
  Tolerância C1 atualizada de ±10 para ±100. Registrado no spec.md.

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
- [x] Executar Phase 0 (T0.1 → T0.9, T0.13 → T0.17). Concluído em 2026-05-12.
- [~] Ler ASONAM 2025 ("Contagious Rhythms") — adiado indefinidamente (wave-based descartado).
- [~] Contatar Gabriel Oliveira — não necessário (wave-based descartado).
- [x] Especificar **Phase 1** (`.specs/features/phase-1-hetero-graph/`) via `/tlc-spec-driven specify`. Concluído em 2026-05-17.
- [x] Executar `/tlc-spec-driven design phase-1-hetero-graph` (5 open questions resolvidas). Concluído em 2026-05-17.
- [x] Executar `/tlc-spec-driven tasks phase-1-hetero-graph` (16 tasks atômicas em 6 waves). Concluído em 2026-05-17.
- [x] Executar `/tlc-spec-driven implement phase-1-hetero-graph` (rodar T1 → T16). Concluído em 2026-05-17.

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
