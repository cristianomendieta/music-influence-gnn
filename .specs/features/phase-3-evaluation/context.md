# Phase 3 — Context (decisões do specify)

Gray areas resolvidas em 2026-06-28. Duas pelo pesquisador, duas delegadas a mim
(com justificativa registrada). Alimentam `spec.md` → "Decisões fixadas".

## D1 — Escopo P1 = Quantitativo + Qualitativo (pesquisador)

**Pergunta:** com ~2 semanas até o deadline (2026-07-11) e a Phase 4 ainda depois,
o que é must-ship?

**Decisão:** **Quantitativo + qualitativo** como P1. Modo 1 (fit retroativo vs SIR) +
Modo 2 (predição genuína vs SIR) **e** as Figs. 8/9 (casos nomeados) são obrigatórios.
A **análise interpretativa** (ablation, importância de features, análogos β/γ/R₀) é **P2**
— entregue se o prazo permitir, sem ser gate de conclusão.

**Como aplicar:** C1–C8 + C12 são gates; C9–C11 são P2.

## D2 — Granularidade = semanal (delegada → minha sugestão)

**Pergunta:** GNN prediz semanal (`y_week`), SIR foi ajustado no diário. Como tornar o
RMSE comparável 1-pra-1?

**Sugestão adotada:** **agregar tudo para semanal.**
**Por quê:** o GNN é intrinsecamente semanal (a máscara do grafo é por `first_seen_week`);
fazer upsample semanal→diário injetaria artefato de interpolação que penaliza o GNN com
erro que não é do modelo. O SIR é regenerável dos parâmetros e agregável a média por
ISO-week — barato e exato. As métricas **diárias** do Phase 0 ficam preservadas como
âncora de reprodução-do-paper; o head-to-head justo é o **semanal**.
**Ressalva (declarar no paper):** a suavização semanal reduz a estrutura multi-onda que
prejudicava o SIR em hits longos — pode *favorecer* o SIR nesse subgrupo. Por isso o
recorte de hits longos (R1.6) é obrigatório, para expor o efeito.

## D3 — Horizontes Modo 2 = k ∈ {1,2,4} semanas via rollout recursivo (delegada → minha sugestão)

**Pergunta:** o modelo é forecasting de 1 passo; como definir os horizontes da predição
genuína (paper usava k ∈ {1,7,14,30 dias})?

**Sugestão adotada:** **rollout recursivo, k ∈ {1,2,4} semanas** (≈ 7/14/30 dias do paper).
**Por quê:** é o caminho natural e barato para um forecaster de 1 passo; cabeça-direta-por-
horizonte exigiria re-treinar o grid 4× (inviável em 2 semanas). No rollout, a feature de
popularidade defasada (`pop_bank`) passa a usar valores **preditos** além da origem — é
isso que torna a predição genuína honesta (sem vazamento). Critério de sucesso ajustado de
"≥2 de 4" para **≥2 de 3** horizontes.
**Ressalva:** o rollout acumula erro; reportar a degradação por `k` é parte do resultado.

## D4 — Wave-based dropado (pesquisador)

**Decisão:** **comparar só vs SIR.** O wave-based foi descartado na Phase 0 (custo
proibitivo, 2026-05-12) e fica fora do escopo da Phase 3. A ausência é **declarada como
limitação** no paper. O critério "resultado forte = bater wave-based" sai de cena;
permanecem os critérios mínimo (vs SIR) e o piso publicável.

---

## Achados de disco (2026-06-28) que reforçam R0

Verificado no repo nesta sessão:

- `results/*` é **gitignored** (só `.gitkeep` versionado). `results/phase0/` **não existe**
  nesta máquina — `sir_params.parquet`, `summary.md`, `boxplot_fig3.png` ausentes.
  → Phase 3 **regenera** o SIR via `scripts/run_phase0.py` / `baselines/sir.py`.
- `data/processed/subset_ids.json` **gitignored e ausente**; regenerável via `build_subset`
  (`src/music_diffusion_gnn/data/subset.py`).
- Versionados (whitelist do `.gitignore`): `data/processed/timeseries.parquet` (diário,
  cols `song_id,chart,date,rank_score,y`, 4,44M linhas), `graph/hetero_full.pt`,
  `graph/node_id_map.json`.
- Phase 2 (Colab) presente em `results/phase2_experimentos_v2/`, mas o `predictions.parquet`
  é da config fraca **W4_h64_l2_lr1e-03** ("run único"), **não** da melhor **W12_h128_l3**.
  → Phase 3 **regenera as predições** na melhor config (OQ2: pinar o checkpoint).

## Insight metodológico levantado no specify (vira OQ1, crítica)

A cabeça residual do GNN (`ŷ = clamp(y_prev + Δ)`) implica que, sob **teacher forcing**,
o modelo sempre recebe o `y(w-1)` **real** → a "reconstrução retroativa" vira quase-
persistência e venceria o SIR **trivialmente** (o SIR não tem acesso ao valor defasado).
Não é comparação justa. Resolução recomendada (a confirmar no design): **Modo 1 = rollout
livre** a partir de uma janela-seed (análogo direto ao SIR *fit-then-simulate*); o
teacher-forced de 1 passo é exatamente o **Modo 2 `k=1`**. Isso muda o *significado* do
critério mínimo C4 — só é informativo se o Modo 1 for rollout livre.
