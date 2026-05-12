# PROJECT — Music Diffusion GNN

> Operational summary. Fonte detalhada da pesquisa: [`PLANO.md`](../../PLANO.md).

## Vision

Mostrar que **sinais relacionais** (colaborações entre artistas, vizinhança de gênero,
co-trajetória entre músicas no chart) capturam variância da popularidade musical
brasileira que **modelos populacionais** (SIR clássico, wave-based) deixam de fora.

A linha de pesquisa de Oliveira et al. (BraSNAM 2025, ASONAM 2025, IEEE Access 2025)
modela cada música isoladamente como uma curva epidêmica agregada. Nenhum desses
trabalhos incorpora estrutura artista–música–gênero. Esse é o gap.

## Research question

Um Temporal GNN heterogêneo treinado sobre o grafo artista–música–gênero do MGD+
consegue (a) reduzir o erro de fit em hits de longa duração — onde o SIR é fraco
(RMSE 2× maior que virality no paper original), (b) prever melhor com horizonte
k > 1, e (c) manter parâmetros estruturalmente interpretáveis (atenção por tipo
de aresta, importância de features de gênero/artista)?

## Contributions (3 bullets do paper)

1. **Reformulação** do problema de difusão musical do MGD+ como aprendizado em
   grafo temporal heterogêneo. Schema explícito artista–música–gênero com
   co-trajetória direcionada.
2. **Comparação justa** contra dois baselines populacionais (SIR clássico,
   wave-based) em dois regimes (fit retroativo + predição genuína), com o mesmo
   dataset, pré-processamento e métricas.
3. **Análise interpretativa**: pesos de atenção por tipo de aresta, features
   mais informativas, casos de falha do GNN.

## Goals

- **Workshop alvo**: BraSNAM 2026 (mesma comunidade do paper original).
- **Formato**: short paper SBC, 8–12 páginas.
- **Janela**: 10 semanas a partir de 2026-05-02.
- **Submissão alvo**: ~2026-07-11 (semana 10).

## Success criteria

**Mínimo aceitável para submeter:**
- GNN bate SIR clássico em RMSE de success (Modo 1 — fit retroativo) com
  p < 0,01 no teste de Mann-Whitney.
- GNN bate SIR clássico em ≥2 dos 4 horizontes do Modo 2 (predição genuína:
  k ∈ {1, 7, 14, 30 dias}).

**Resultado forte:**
- GNN bate **wave-based** (Oliveira ASONAM 2025) em success (Modo 1).
- Redução de RMSE ≥30% em hits longos (subset com duração > 90 dias no chart).
- Análise interpretativa mostra atenção significativa em arestas
  artista→música e gênero↔gênero.

**Resultado limitado mas publicável:**
- GNN não bate wave-based, mas mostra ganhos em subgrupo específico (por gênero
  ou por duração). Reposicionar como "competitivo com modelos populacionais
  sofisticados em uma fração relevante das músicas, com a vantagem de
  incorporar sinais relacionais".

## Constraints & limitations

- **Período**: 2017-01-01 a 2021-12-31 (1.826 dias). Paper original cobre até
  2022-03-13 — diferença de ~2,5 meses (4% do período total). **Declarar como
  limitação no paper, não esconder.**
- **Subset principal de análise**: 1.179 músicas (interseção viral ∩ hit).
  Paper original: 1.977 — diferença explicada pelo gap de período.
- **Hardware**: laptop com GPU é suficiente. Modelo alvo (~200K params) treina
  em horas, não dias.

## Out of scope

- Predição zero-shot para músicas que nunca entraram no chart.
- Modelagem multi-país (apenas Brasil).
- Recomendação musical (objetivo é modelar difusão, não recomendar).
- Comparações com short-form video / TikTok (fica para a Proposta 1 do mestrado).

## Stack

```
pandas, numpy             # manipulação
scipy                     # fit SIR + wave-based
networkx                  # construção e análise do grafo
torch, torch-geometric    # HeteroData, HeteroConv, SAGEConv
pytorch-lightning         # loop de treino
wandb                     # tracking
seaborn, matplotlib       # figuras do paper
```

## References

- Oliveira et al. 2025 (BraSNAM) — paper sendo replicado/estendido.
- Oliveira et al. 2025 (ASONAM, "Contagious Rhythms") — wave-based, baseline obrigatório.
- Oliveira et al. 2025 (IEEE Access) — causalidade virality↔success.
- Seufitelli et al. 2023 (DSW) — paper do dataset MGD+.
- Hamilton et al. 2017 — GraphSAGE.
- Hu et al. 2020 — HGT (alternativa de arquitetura).
- Rossi et al. 2020 — TGN (alternativa de arquitetura).
- Kempe et al. 2003 — IC e LT (modelos clássicos de difusão em rede).
