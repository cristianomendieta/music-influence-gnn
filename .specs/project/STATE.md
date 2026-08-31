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

- **2026-05-30** — **Phase 2 especificada** (`.specs/features/phase-2-temporal-gnn/spec.md` + `context.md`).
  3 gray areas resolvidas no specify: (D1) objetivo de treino = **Ambos** (forecasting 1-passo
  para predição genuína + fit retroativo da curva para comparação com SIR; encoder compartilhado);
  (D2) barra de conclusão = **bater persistência ingênua** `ŷ(t)=y(t-1)` no val MSE em ambos os
  regimes — comparação rigorosa vs SIR deferida à Phase 3; (D3) HPs = **grid pequeno do ROADMAP**
  (W∈{4,8,12}, hidden∈{64,128}, layers∈{2,3}, lr∈{1e-3,5e-4}), seleção por val MSE.
  Alvo confirmado: `y(t)∈[0,0.5]` do `timeseries.parquet` (mesma def. do Phase 0), regimes
  virality (viral50) / success (top200). 6 open questions p/ design (tratamento de semanas
  fora do chart, cache de embeddings por semana, padding de janela, def. exata do fit retroativo,
  edge subsampling, batching causal).

- **2026-05-30** — **Phase 2 design aprovado** (`.specs/features/phase-2-temporal-gnn/design.md`).
  **Descoberta decisiva:** `timeseries.parquet` é **diário** (4,44M linhas) mas o grafo é **semanal**
  (`first_seen_week∈[0,260]`, `mask_until` por semana). Logo o alvo é modelado em **granularidade
  semanal** (`y_week` = média diária por ISO-week); linhas de 2022 (`week>260`) descartadas.
  Comparabilidade semanal↔diária com SIR fica para a Phase 3 (decisão T-gran).
  OQ1–OQ6 resolvidas: (OQ1) treinar no span ativo `[first_seen..last_seen]` incl. semanas de baixa
  popularidade; (OQ2) **banco de embeddings por semana computado por minibatch** (só semanas distintas
  do batch, cache intra-forward, backprop através dos snapshots — não cacheável entre épocas pois pesos
  mudam); (OQ3) left-pad zeros + pad_mask, alvo só p/ `w>first_seen`; (OQ4) **um modelo, dois
  protocolos**: forecasting (held-out) + retroativo (reconstrução in-sample teacher-forced);
  (OQ5) edge subsampling **deferido** (grafo completo ~700K arestas OK em CPU; dropout+wd+early-stop);
  (OQ6) minibatch de tuplas `(song,chart,week)` com shuffle livre (janela causal própria).
  Componentes: `models/{encoder,temporal_head,diffusion_gnn,baselines}.py`, `training/{dataset,trainer}.py`,
  `scripts/run_phase2.py`. `models/` e `training/` estão vazios (só `__init__`).
  **Dívida técnica:** `pyarrow` ausente na `.venv` (a `.venv` aponta interpreter de path antigo
  `music-diffusion-gnn`; `.venv/bin/pip` quebrado, usar `.venv/bin/python -m pip`); pyarrow instalado
  nesta sessão, falta fixar em `pyproject.toml`.

- **2026-05-30** — **Phase 2 tasks definidas** (`.specs/features/phase-2-temporal-gnn/tasks.md`).
  15 tasks atômicas em 6 waves: T1 deps → T2/T3/T4 dataset (sequencial) → T5/T6/T8 modelos (paralelos)
  + T7 diffusion_gnn (após T5,T6) → T9/T10/T11 trainer (sequencial) → T12/T13 testes (paralelos) →
  T14 run_phase2 → T15 execução real. Cada task = 1 commit; PR único final. Sem MCPs/Skills (PyG puro).
  Confirmado: `pyarrow` **não** está em `pyproject.toml` (deps linhas 16–27) — T1 = adicionar `pyarrow>=14.0`.
  Testes em `tests/` (raiz, padrão Phase 1: `test_phase1_build.py`, `test_phase1_temporal.py`).
  Artefatos confirmados: `data/processed/graph/{hetero_full.pt,node_id_map.json}`, `data/processed/subset_ids.json`.

- **2026-05-31** — **Phase 2 implementada** (T1–T14 concluídas; T15 grid rodando).
  Componentes implementados: `models/{encoder,temporal_head,diffusion_gnn,baselines}.py`,
  `training/{dataset,trainer}.py`, `scripts/run_phase2.py`, `tests/test_phase2_{leakage,forward}.py`.
  **Descobertas de implementação:**
  (I1) `week_index` não é bijetivo para anos com 53 semanas ISO (2020-W53 e 2021-W01 → índice 208);
  corrigido usando partição estrita: train≤182, val∈(182,208), test≥208.
  (I2) Granularidade semanal implementada via `pandas.Series.dt.isocalendar()` vetorizado
  (chamada por linha ao `week_index` causava ValueError em 2017-01-01 = ISO 2016-W52).
  (I3) OQ5 (edge subsampling) **desadiado**: 664K arestas cotrajetória esgotavam memória
  autograd em WSL. Solução: `max_cotraj_edges=30_000` (DropEdge ~4,5% das arestas por snapshot).
  (I4) Batching por semana-alvo (`_iter_batches` week-grouped): amostras agrupadas por `target_week`
  → banco compartilhado na semana (`retain_graph` ou sub-batch único). 14× speedup vs shuffle livre.
  (I5) `predict()` vetorizado: `bank[wk][song_idxs]` (fancy indexing) em vez de loop B×W.
  Benchmark final: 38s/época (W=4, dataset completo 321K amostras, 1981 músicas).
  Grid 24 configs × ~30 épocas estimado em 10–15h (rodando em background, PID 292329).
  C6/C7 pendentes (GNN não bateu persistência no smoke test com 5 épocas/50 músicas — esperado).

- **2026-06-23** — **Phase 2 v1 REPROVOU C6/C7.** Grid completo (24 configs, via notebook em
  Colab T4) rodou; **nenhuma config supera a persistência ingênua** `ŷ(w)=y(w-1)`. Melhor da grid
  `W12_h128_l3_lr5e-04` val_mse≈0.00506 vs persistência≈0.0009 (~5× pior). `summary.md` foi gerado
  numa config fraca (`W4_h64_l2_lr1e-03`, 16ª/24). **Causa-raiz (estrutural):** o modelo nunca recebia
  o alvo defasado `y(w-1)` — entrada da GRU era só o embedding estrutural; features de nó música são
  acústicas estáticas. Logo errava o *nível* da série. Plano B do ROADMAP (HGT/Transformer) não
  resolveria (é problema de feature, não de capacidade).

- **2026-06-23** — **Phase 2 Revisão R1 especificada + implementada.** Decisão: injetar
  popularidade defasada. R1-D1 **feature de nó dinâmica** `pop_bank[w]` (2 canais viral50/top200)
  antes do HeteroSAGE → popularidade difunde pela rede de influência (estrutura load-bearing).
  R1-D2 **cabeça residual** `ŷ=clamp(y_prev+Δ,0,0.5)`, Δ=GRU+MLP. R1-D3 **zero-init** da última
  Linear → no init Δ=0 → reproduz a persistência exatamente. R1-D4 `y_prev` lido do `pop_bank[w-1]`
  (= valor da persistência; **não** muda `build_samples`). Sem leakage (`w'≤w-1`). Detalhes em
  `design.md` → Revisão R1 e `tasks.md` → Wave R1 (R1.T1–R1.T8).
  **Componentes alterados:** `training/dataset.py` (+`build_pop_bank`), `models/temporal_head.py`
  (Δ cru + zero-init), `models/diffusion_gnn.py` (`pop_bank` buffer + injeção + resíduo),
  `training/trainer.py` (repassa `pop_bank`), `scripts/run_phase2.py`, `tests/test_phase2_forward.py`
  (+`test_residual_starts_at_persistence`, +`test_pop_injection_forward_runs`), notebook.
  **Smoke (gate R1.T7) PASSOU:** subset 60 músicas, 7 épocas CPU → GNN **bate** persistência nos
  dois regimes (viral50 0.000684 vs 0.000722; top200 0.000537 vs 0.000540). 8/8 testes phase2 verdes.
  Pendente: re-rodar o grid completo (R1.T8) e conferir C6/C7 no dataset cheio.

- **2026-06-28** — **R1.T8 concluída — C6/C7 APROVADOS no dataset completo.** Grid v2 (24 configs,
  Colab T4) em `results/phase2_experimentos_v2/`. **Todas as 24 configs batem a persistência** e
  ficam fortemente agrupadas (val_mse combinado 0.000749–0.000764 vs ~0.005–0.006 da v1 → ~6,7×
  melhor; persistência combinada ~0.0009). Melhor config = **W12_h128_l3_lr5e-04** (val_mse 0.000749),
  mesma da v1. Run detalhado (`summary.md`, config W4_h64_l2_lr1e-03):
  · VAL forecasting: viral50 GNN 0.000834 < persist 0.000964 ✓; top200 0.000634 < 0.000861 ✓
  · TEST forecasting (semana≥208): viral50 0.000622 < 0.000716 ✓; top200 0.000494 < 0.000618 ✓
  **Phase 2 concluída.** Nota: como todas as configs beat persist e diferem pouco, a escolha de HP
  é robusta; o resíduo ancora e a estrutura agrega ~18% sobre a persistência. Detalhe a polir p/ o
  paper: rodar a avaliação detalhada na MELHOR config (W12_h128_l3), não na W4 do `summary.md`.

- **2026-06-28** — **Phase 3 especificada** (`.specs/features/phase-3-evaluation/spec.md` + `context.md`).
  4 gray areas resolvidas: (D1) escopo P1 = **quantitativo + qualitativo** (Modo 1 + Modo 2 vs SIR
  + Figs 8/9; interpretativo = P2); (D2) granularidade = **semanal** (SIR regenerado e agregado a
  ISO-week; diário do Phase 0 vira âncora de reprodução; ressalva: semanal favorece SIR em hits
  longos → recorte obrigatório); (D3) horizontes Modo 2 = **k∈{1,2,4} semanas via rollout recursivo**
  (pop_bank usa valores preditos além da origem; critério ajustado p/ **≥2 de 3**); (D4) **wave-based
  dropado**, só vs SIR, ausência declarada como limitação.
  **Achados de disco decisivos:** `results/*` é **gitignored** → `results/phase0/sir_params.parquet`
  e `data/processed/subset_ids.json` **NÃO estão nesta máquina**; ambos regeneráveis (R0:
  `scripts/run_phase0.py` + `build_subset`). O `predictions.parquet` em disco é da config fraca **W4**
  ("run único"), não da melhor **W12_h128_l3** → Phase 3 regenera predições na melhor config.
  **OQ1 crítica (fairness Modo 1):** a cabeça residual `ŷ=clamp(y_prev+Δ)` sob teacher forcing vê
  o `y(w-1)` real → reconstrução vira quase-persistência e venceria o SIR trivialmente. Resolução
  recomendada: Modo 1 = **rollout livre** a partir de janela-seed (análogo ao SIR fit-then-simulate);
  teacher-forced de 1 passo = Modo 2 k=1. 6 open questions p/ design (OQ1–OQ6).

- **2026-06-28** — **Phase 3 design aprovado** (`.specs/features/phase-3-evaluation/design.md`).
  6 OQs resolvidas com **sondagens reais** (não suposições): (OQ1 fairness) Modo 1 = **rollout
  livre global sincronizado** — muta `pop_bank` de trabalho e re-encoda; `seed_weeks=W=12`;
  encode **1×/semana** (≈ um passe, minutos em CPU), não O(músicas×span×W); (OQ2) checkpoint =
  **`grid_best_model.pt`** (wrapper c/ meta; é a W12_h128_l3, state_dict inclui `pop_bank`) —
  **`best_model.pt` é a W4 fraca, não usar**; carregar via `strict=False` dropando `pop_bank` e
  usando o regenerado (guard `allclose`); (OQ3) SIR causal = refit `≤w` + simulate, restrito ao
  **test span (208–260)**, `min_hist_weeks=4`, não-converg. excluída e contada; (OQ4) **Wilcoxon
  pareado** primário + Mann-Whitney secundário + **bootstrap IC95% B=10k**; (OQ5) **hit longo =
  >90d com `rank_score>0`** (viral50 77/4%, top200 798/40%) — span denso/floor capturaria 97%,
  armadilha evitada; (OQ6) CRPS **deferido** (GNN determinístico). **Achados de disco decisivos:**
  `timeseries.parquet` é **100% denso** com `y` pisado em 0.001 fora do chart (`rank_score==0`),
  92,7% das linhas ≤0.001 → "dias no chart" = `rank_score>0`. Região RMSE M1 = span completo
  (primário, alinha fit SIR Phase 0) + on-chart (robustez, blinda contra "só prever o floor").
  **Componentes novos** (em `src/.../evaluation/`): `rollout.py`, `sir_eval.py`, `stats.py`,
  `longhits.py`, `figures.py`, `interpretability.py` (P2) + `scripts/run_phase3.py`. Reusa
  pesado: `MusicDiffusionGNN.encode_weeks/predict/pop_bank`, dataset (aggregate_weekly/
  build_samples/build_pop_bank), `fit_sir`/`_sir_curve`/`parallel.fit_all`, `metrics.py`,
  `report.make_boxplot`, `run_phase0.main` (R0). Próximo: `tasks phase-3-evaluation`.

- **2026-06-28** — **Phase 3 tasks definidas** (`.specs/features/phase-3-evaluation/tasks.md`).
  **12 tasks atômicas em 5 waves.** Wave 0 (paralelo): T1 `model_io.load_grid_best_model` (ckpt W12
  + guard pop_bank), T2 `longhits`, T3 `stats` (Wilcoxon/bootstrap/dir.acc.), T4 `persistence_multistep`.
  Wave 1: T5→T6 `sir_eval` (M1 curva semanal do fit; M2 refit causal `≤w`), T7→T8 `rollout` (M1 livre
  global sincronizado; M2 recursivo + **teste de leakage** dedicado). Wave 2 (paralelo): T9 `figures`
  (Fig.3 + Figs.8/9), T10 `interpretability` (P2). Wave 3: T11 `run_phase3.py` (orquestra R0→M1→M2→
  figuras→summary+checklist C1–C12; smoke gate `--smoke`). Wave 4: T12 execução real + registro.
  Cada task = 1 commit; PR único final. **Sem MCPs/Skills** (stack puro). Política de testes inferida
  (não há TESTING.md): pytest `tests/test_phase3_*.py`; unit co-localizado nas funções puras/rollout,
  smoke nas figuras, integration no orquestrador. **Refinamento nas tasks:** extrair o carregamento
  do checkpoint para `evaluation/model_io.py` (o design o mostrava inline) — função testável isolada,
  mantém `run_phase3` fino. Próximo: `implement phase-3-evaluation`.

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

- **2026-06-23** — Persistência ingênua `ŷ(w)=y(w-1)` é uma baseline **fortíssima** em séries
  semanais suaves e autocorrelacionadas. Um forecaster temporal que não recebe `y(w-1)` como
  entrada (direta ou via feature de nó) tende a perder feio (erra o nível). Lição de arquitetura:
  ancorar a predição na persistência (parametrização **residual**) e dar acesso ao histórico de
  popularidade — daí a estrutura só precisa aprender a *correção*.

- **2026-06-23** — **GPU local (GTX 1050 Ti, sm_61) é incompatível** com o PyTorch instalado na
  `.venv` (cu130, suporta sm_75+). Treino local cai para **CPU** (smoke: ~33s/época, subset 60).
  O notebook é feito para **Colab T4 (sm_75)**, onde a GPU funciona. Para grid local completo,
  contar com CPU (horas) ou instalar uma wheel de torch compatível com sm_61.

## Todos

- [x] Especificar Phase 0 (`.specs/features/phase-0-baselines/`).
- [x] Executar Phase 0 (T0.1 → T0.9, T0.13 → T0.17). Concluído em 2026-05-12.
- [~] Ler ASONAM 2025 ("Contagious Rhythms") — adiado indefinidamente (wave-based descartado).
- [~] Contatar Gabriel Oliveira — não necessário (wave-based descartado).
- [x] Especificar **Phase 1** (`.specs/features/phase-1-hetero-graph/`) via `/tlc-spec-driven specify`. Concluído em 2026-05-17.
- [x] Executar `/tlc-spec-driven design phase-1-hetero-graph` (5 open questions resolvidas). Concluído em 2026-05-17.
- [x] Executar `/tlc-spec-driven tasks phase-1-hetero-graph` (16 tasks atômicas em 6 waves). Concluído em 2026-05-17.
- [x] Executar `/tlc-spec-driven implement phase-1-hetero-graph` (rodar T1 → T16). Concluído em 2026-05-17.
- [x] Especificar **Phase 2** (`.specs/features/phase-2-temporal-gnn/`). Concluído em 2026-05-30 (3 gray areas resolvidas).
- [ ] Executar `/tlc-spec-driven design phase-2-temporal-gnn` (resolver OQ1–OQ6: semanas off-chart, cache de embeddings, padding, fit retroativo, edge subsampling, batching causal).
- [x] Executar `/tlc-spec-driven tasks phase-2-temporal-gnn` (15 tasks atômicas em 6 waves). Concluído em 2026-05-30.
- [~] Executar `/tlc-spec-driven implement phase-2-temporal-gnn` — T1–T14 concluídas 2026-05-31; T15 (grid) rodando.
- [x] Conferir C1–C9 quando grid terminar — grid v1 reprovou C6/C7 (GNN perde p/ persistência). 2026-06-23.
- [~] Plano B (HGT/Transformer) **não acionado**: causa-raiz é feature (cego a `y(w-1)`), não capacidade. Substituído pela Revisão R1.
- [x] **Revisão R1** especificada + implementada (R1.T1–R1.T6) + smoke (R1.T7) passou. 2026-06-23.
- [x] **R1.T8**: grid v2 rodado no Colab (T4); C6/C7 APROVADOS. Registrado 2026-06-28.
- [ ] (Polish opcional) Rodar avaliação detalhada na melhor config (W12_h128_l3) p/ os números do paper.
- [x] **Phase 3** — especificada (`.specs/features/phase-3-evaluation/`, spec + context). 2026-06-28.
- [x] Executar `/tlc-spec-driven design phase-3-evaluation` (OQ1–OQ6 resolvidas; design.md aprovado). 2026-06-28.
- [x] Executar `/tlc-spec-driven tasks phase-3-evaluation` (12 tasks atômicas em 5 waves). 2026-06-28.
- [ ] Executar `/tlc-spec-driven implement phase-3-evaluation` (rodar T1 → T12).
- [x] **Item 21** — sonda do canal de gênero (pesos iniciais + treinados): canal utilizável. 2026-08-31.
- [x] **Item 05** — gênero estrutural + re-treino no regime `current`. 2026-08-31.
- [ ] **Item 06** — completar a matriz: `current` seeds 43/44 e `pre_pandemia` seeds 42/43/44 (5 treinos).
- [ ] **Item 09** — refazer a ablação por tipo de aresta sobre o checkpoint novo, `cooccurs` incluído,
      mais a sonda de `has_genre`/`rev_has_genre` herdada do item 21.

## Consolidação documental e novo marco (2026-08-18)

Qualificação aprovada. Sessão de grilling definiu o marco "a estrutura relacional
agrega?" — decisões em `docs/adr/0001` a `0005`, plano em `ROADMAP.md`, vocabulário
em `CONTEXT.md` (raiz do repo).

**Documentos removidos por duplicidade:** `STATUS.md` (parado na Phase 0/1) e
`EXPERIMENTS.md` (só Phase 0, e com RMSE 0,0381/0,0699 que contradiziam os
0,0289/0,0471 de `results/phase0/summary.md`, os canônicos). Também removido
`documento_qualificao.zip`, duplicata da pasta `documento_qualificao/`.

**Fontes de verdade a partir daqui:** `README.md` (entrada), `.specs/project/STATE.md`
(estado), `.specs/project/ROADMAP.md` (plano), `CONTEXT.md` (glossário), `docs/adr/`
(decisões), `PLANO.md` (visão de pesquisa e posicionamento na literatura).

**Faxina em `results/`.** Havia quatro pastas de Phase 3 sobrepostas. A canônica é a
de 2026-07-07, que tem a rodada completa (1.955 músicas) **e** a interpretabilidade;
foi renomeada para `results/phase3/`. Removidas: a antiga `phase3` (07-04, sem
interpretabilidade), `phase3_resultados` (smoke de 5 músicas) e as pastas de Phase 2
anteriores à correção de vazamento de 2026-06-14 (`phase2/`, `phase2_experimentos/`).
Mantida `results/phase2_experimentos_v2/`, que tem o checkpoint em uso.

**Achado P0 aberto.** A interpretabilidade C9–C11 **não** estava pendente: rodou em
2026-07-07 e devolveu `delta_rmse` exatamente zero para os cinco tipos de aresta e
para os três grupos de features. Ver Phase 4 do ROADMAP e ADR-0004. É o bloqueador
de maior prioridade: ou o instrumento satura no `clamp`, ou o grafo não influencia a
predição.


## Diagnóstico da ablação zerada — fechado (2026-08-30)

**Desfecho A, com hipótese C confirmada.** O `delta_rmse` exatamente zero de 2026-07-07
tinha duas causas somadas, e a dominante era instrumental: `_predict_all` encodava a
semana alvo, que `MusicDiffusionGNN.predict` nunca lê (ela não está na janela
`[w−W, …, w−1]`). 0% das posições da janela chegavam ao GRU, a sequência era nula e `Δ`
saía constante em 0,00320397 para as 98.186 amostras. Corrigido em `interpretability.py`
(commit `2c645c6`). Sobre isso, o `clamp` satura: anula 76,9% da correção na leitura
completa contra 3,6% no recorte on-chart.

Relatório com todas as tabelas: [`docs/diagnostico-ablacao.md`](../../docs/diagnostico-ablacao.md).
Artefatos brutos em `MyDrive/music-influence-gnn/item04_diagnostico/` (fora do git).

**Achado P0 fechado.** `results/phase3/interpretability.parquet` mede uma constante:
não é resultado, é artefato, e não entra na dissertação. A permutação por grupo de
features tem o mesmo defeito.

**Três achados novos, abertos:**

- **Cotrajetória carrega todo o sinal.** Removê-la leva o RMSE on-chart de 0,1005 para
  0,1938 (+93%). Os outros quatro tipos de aresta têm `delta_rmse` negativo: removê-los
  **melhora** o modelo.
- **Canal de gênero inerte (P1) — causa isolada em 2026-08-30.** Esvaziar as 9.866
  arestas `cooccurs` não alterava nenhum embedding de `music` além do ruído de float
  (3e−08). A sonda camada a camada mostrou que **não era falta de caminho**: era a
  agregação. `SAGEConv` usa a média dos vizinhos, e a média da tabela de 530×32
  parâmetros livres, i.i.d. centrada em zero, se cancela. Com os atributos estruturais
  do ADR-0003 o sinal que chega em `music` fica ~83× maior. Ver
  [`docs/sonda-canal-genero.md`](../../docs/sonda-canal-genero.md). Falta repetir a
  medição com pesos treinados (item 21, seção 4 do notebook do item 05).
- **Descasamento de densidade treino/avaliação (P1, afeta a Phase 6).** Treino com
  `max_cotraj_edges = 30_000` por snapshot (`trainer.py:48`), avaliação com o grafo
  completo (480k–664k arestas). Religar a cotrajetória ao acaso melhora o erro on-chart
  (−0,0028), o que é adverso à tese mas confundido por esse descasamento: para o modelo,
  um religamento aleatório se parece mais com o que ele viu no treino do que o grafo
  completo. O protocolo da escada precisa fixar o mesmo orçamento nos dois lados.


## Phase 5 — Gênero estrutural (item 05): código entregue em 2026-08-30

Gênero deixou de ser tabela aprendida de 530×32 e passou a quatro atributos com fórmula
derivados da rede gênero↔gênero ([ADR-0003](../../docs/adr/0003-atributos-de-genero-derivados.md)):
`degree`, `weighted_degree`, `n_artists` (log1p + escore-z) e a bandeira
`absent_from_network`. Fórmula em `graph/nodes.py::genre_attributes`.

**Restrição causal implementada:** só os anos inteiramente contidos na janela de treino
entram (`SplitRegime.train_years` → `current` 2017-2019, `pre_pandemia` 2017-2018), então
**cada regime tem seu grafo**: `data/processed/graph/hetero_full_{regime}.pt`, resolvido
por `graph.build.graph_path(regime)`. `hetero_full.pt` deixou de existir. `Avg_Popularity`
e `Avg_Streams` não são lidas em lugar nenhum — saíram também do `edge_attr` de `cooccurs`,
que passou de 4 para 2 colunas.

C1–C9 verdes nos dois regimes. Suíte de testes verde (`tests/test_genre_features.py` novo).

**Consequência:** todo checkpoint anterior a esta mudança é inválido — carrega
`genre_emb.weight` e um encoder dimensionado para o gênero antigo. `model_io` recusa esses
arquivos com mensagem explícita, e `tests/test_phase3_model_io.py` fica em skip até o
re-treino.

**Re-treino e sonda treinada: concluídos em 2026-08-31** (regime `current`, seção abaixo).

## Phase 5 — re-treino e canal de gênero: fechado (2026-08-31)

Rodado no Colab GPU pelo [`notebooks/item05_genero_estrutural_colab.ipynb`](../../notebooks/item05_genero_estrutural_colab.ipynb)
(commit `4a3e7e7`), regime `current`, seed 42, config `W12_h128_l3_lr5e-04`, 35 épocas em 21,4 min.
Relatório: [`docs/genero-estrutural-retreino.md`](../../docs/genero-estrutural-retreino.md).
Artefatos em `results/item05_genero_estrutural/` (gitignored) e no Drive.

**1. A nova representação de gênero não custa desempenho.** `val_mse` 0,000754 contra 0,000749
do grafo antigo (+0,7%), com a comparação pareada na configuração e nas features de música e
artista — a remoção de vazamento que as levou a 12 e 1 dimensões é de 2026-06-14, anterior à
grid v2. A diferença cai dentro da dispersão da própria grid v2 (0,000749–0,000764 em 24
configs). Uma seed só: não é empate estatístico ainda, mas é evidência de que trocar 16.960
parâmetros livres por quatro colunas com fórmula não degradou nada. A crítica da banca fica
atendida sem preço.

**2. C6/C7 continuam verdes.** Forecasting de 1 passo, leitura completa: val viral50 0,000839 <
0,000964 da persistência; val top200 0,000619 < 0,000861; teste viral50 0,000636 < 0,000716;
teste top200 0,000488 < 0,000618. Não comparar célula a célula com o `summary.md` da Phase 2:
aquele saiu da config fraca W4.

**3. Item 21 fechado: canal utilizável.** Com pesos treinados, esvaziar `cooccurs` move o
embedding de `music` em **1,9e−04** contra os **3e−08** do modelo antigo — ~6.200× maior, quatro
ordens acima do ruído de float, 70% dos nós de música alterados. A inércia do item 04 era das
duas causas somadas: a média da tabela aleatória se cancelava **e** o treino não preservava o que
chegava por ali. Tabelas em [`docs/sonda-canal-genero.md`](../../docs/sonda-canal-genero.md).

**Ressalva de escala, que é o achado a carregar adiante.** O sinal de gênero cai de 24,8% da
magnitude típica do embedding em `genre` para 15,4% em `artist` e **0,5%** em `music`, num
embedding 92% esparso pós-ReLU. O canal conduz, mas o `val_mse` não se mexeu: **conduzir não é
ser útil**. A utilidade é o que o item 09 mede.

**Consequência direta para a Phase 6.** A ablação por tipo de aresta precisa ser refeita sobre
este checkpoint. O `delta_rmse` exatamente 0,000000 de `cooccurs` no item 04 foi medido num
modelo com o canal morto; se o zero persistir com o canal vivo, o achado passa a ser sobre a
utilidade do gênero, não sobre a propagação — afirmação bem mais forte e reportável.

**Pendências que migraram para o item 06:** regime `pre_pandemia` (`absent_from_network` sobe de
20,6% para 30,8% dos gêneros, então a representação muda de verdade entre os regimes) e as seeds
43 e 44. 1 dos 6 treinos da matriz já está feito.


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
