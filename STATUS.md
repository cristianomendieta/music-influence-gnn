# Status do Projeto — Music Influence GNN

**Atualizado:** 2026-05-17 | **Submissão alvo:** BraSNAM 2026 (2026-07-11)

## Onde estamos agora

| Fase | Descrição | Status | Semanas |
|------|-----------|--------|---------|
| **Phase 0** | Reprodução baselines SIR | ✅ Concluído | W1–W2 (2026-05-02→12) |
| **Phase 1** | Construção grafo heterogêneo | ⏳ Próxima | W2–W3 |
| **Phase 2** | Temporal GNN (HeteroSAGE + GRU) | — Pendente | W3–W6 |
| **Phase 3** | Avaliação dupla (retroativa + predição) | — Pendente | W6–W8 |
| **Phase 4** | Escrita e submissão | — Pendente | W8–W10 |

## Resultados Phase 0 (baseline SIR)

Todos os 5 critérios de aceitação passaram — ver [`results/phase0/summary.md`](results/phase0/summary.md).

| Métrica | Obtido | Alvo | Status |
|---------|--------|------|--------|
| RMSE virality | 0,0289 | ≈0,028 ±10% | ✅ |
| RMSE success | 0,0471 | ≈0,052 ±10% | ✅ |
| Mann-Whitney p-value | 1,61e-39 | < 0,05 | ✅ |
| Subset (viral∩hit) | 1.981 músicas | ≥1.900 | ✅ |
| Convergência SIR | 100% | ≥99% | ✅ |

> Wave-based baseline descartado por decisão do pesquisador (custo computacional proibitivo,
> SIR suficiente para validação). Ver STATE.md (2026-05-12).

## Próximo passo

```
/tlc-spec-driven specify phase-1-hetero-graph
```

Referências para Phase 1:
- [`exploration/06_gnn_design_sketch.ipynb`](exploration/06_gnn_design_sketch.ipynb) — schema do grafo heterogêneo
- [`PLANO.md §2`](PLANO.md) — nós (Música, Artista, Gênero), arestas, snapshots semanais
- [`.specs/project/ROADMAP.md`](.specs/project/ROADMAP.md) — timeline completo

## Mapa de documentos

| Documento | Para que serve |
|-----------|---------------|
| [`PLANO.md`](PLANO.md) | Visão geral, perguntas de pesquisa, plano de 10 semanas |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Log de execuções e análise de resultados |
| [`README.md`](README.md) | Setup do ambiente e estrutura do repo |
| [`.specs/project/ROADMAP.md`](.specs/project/ROADMAP.md) | Timeline operacional por fase |
| [`.specs/project/STATE.md`](.specs/project/STATE.md) | Decisões, lessons, todos ativos |
| [`.specs/features/phase-0-baselines/`](.specs/features/phase-0-baselines/) | Spec, design e tasks do Phase 0 |
| [`results/phase0/summary.md`](results/phase0/summary.md) | Resultados numéricos do baseline |

## Código implementado (Phase 0)

```
src/music_diffusion_gnn/
├── data/
│   ├── loaders.py      (131 LOC) — carga de CSVs raw
│   ├── preprocess.py   (122 LOC) — rank score → MA-7d → min-max → floor
│   └── subset.py       (63 LOC)  — filtro viral∩hit
├── baselines/
│   ├── sir.py          (86 LOC)  — modelo SIR (odeint + curve_fit)
│   └── parallel.py     (51 LOC)  — paralelização joblib
└── evaluation/
    ├── metrics.py      (39 LOC)  — RMSE, Mann-Whitney
    └── report.py       (116 LOC) — summary.md + boxplot

scripts/run_phase0.py   (106 LOC) — orquestrador idempotente
tests/                  (156 LOC) — 18 casos, todos verdes

# Stubs para fases futuras:
src/music_diffusion_gnn/graph/     (Phase 1)
src/music_diffusion_gnn/models/    (Phase 2)
src/music_diffusion_gnn/training/  (Phase 2)
```
