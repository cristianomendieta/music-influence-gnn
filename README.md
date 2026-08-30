# Music Influence GNN — difusão de popularidade musical em grafo temporal heterogêneo

Dissertação de mestrado. Replica Oliveira et al. (BraSNAM 2025), que modela
popularidade musical como epidemia via SIR, e estende o problema para aprendizado
sobre o grafo heterogêneo música–artista–gênero da cena brasileira no Spotify.

**Pergunta do marco atual:** sinais relacionais capturam variância da popularidade
que os modelos populacionais deixam de fora, ou o ganho observado vem só da
flexibilidade de um modelo neural?

**Qualificação aprovada em ago/2026.** Defesa prevista para jan/2027.

## Por onde começar

| Documento | O que é |
|---|---|
| [CONTEXT.md](CONTEXT.md) | Glossário. A linguagem do projeto, e só isso |
| [ROADMAP.md](ROADMAP.md) | Resumo de uma página: a pergunta do marco, as decisões e as fases |
| [.specs/project/ROADMAP.md](.specs/project/ROADMAP.md) | O plano detalhado, fase a fase, com a matriz experimental |
| [.specs/project/STATE.md](.specs/project/STATE.md) | Estado, decisões e pendências ao longo do tempo |
| [docs/adr/](docs/adr/) | Decisões difíceis de reverter, com o porquê |
| [PLANO.md](PLANO.md) | Visão de pesquisa e posicionamento na literatura |
| [documento_qualificao/](documento_qualificao/) | Fonte LaTeX da dissertação |

## Onde o trabalho está

Fases 0 a 3 concluídas: baseline SIR reproduzido, grafo construído e validado,
MusicDiffusionGNN treinado, avaliação dupla contra SIR e persistência. Os números
estão em `results/` e no capítulo de resultados.

O marco atual submete a tese a teste. Entram na comparação um baseline neural sem
grafo, a mesma GNN sobre um grafo com arestas embaralhadas, um segundo split
pré-pandemia, três seeds por modelo e o recorte on-chart como leitura principal.
O bloqueador que segurava o marco foi fechado em 2026-08-30: a ablação por tipo de
aresta devolvia variação de erro exatamente zero porque o harness nunca entregava os
embeddings ao modelo, e o `clamp` satura por cima disso na leitura completa. Números e
consequências em [docs/diagnostico-ablacao.md](docs/diagnostico-ablacao.md).

## Estrutura

```
src/music_diffusion_gnn/
  data/         loaders e pré-processamento (posição no chart → média móvel → normalização)
  baselines/    SIR clássico
  graph/        construção do HeteroData
  models/       encoder heterogêneo, cabeça temporal, persistência
  training/     dataset, splits temporais, trainer
  evaluation/   métricas, rollout, estatística, interpretabilidade, figuras

scripts/        run_phase0.py … run_phase3.py (entry points por fase)
notebooks/      pipelines de treino e avaliação rodados no Colab
exploration/    EDA anterior à construção do grafo
tests/          suíte pytest, incluindo testes de vazamento temporal
data/           gitignored, exceto os três artefatos processados que o Colab clona
results/        gitignored; phase0, phase1, phase2_experimentos_v2, phase3
```

## Setup

```bash
pip install -e .[dev]
pytest tests/ -v
```

O treino roda no Google Colab com GPU, não localmente: a máquina de
desenvolvimento é CPU-only. Os notebooks persistem checkpoints no Drive e retomam
a grid de onde parou em caso de desconexão.

## Citações obrigatórias

- **Paper replicado:** Oliveira, G. P.; Vassio, L.; Couto da Silva, A. P.; Moro, M. M.
  *Modeling music popularity as an epidemic: insights from the Brazilian market.* BraSNAM 2025.
- **Dataset:** Seufitelli, D. B.; Oliveira, G. P.; Silva, M. O.; Moro, M. M.
  *MGD+: An Enhanced Music Genre Dataset with Success-based Networks.* DSW 2023.
- **Follow-up dos mesmos autores:** Oliveira, G. P. et al. *Contagious Rhythms: A Wave-Based
  Epidemic Approach for Music Virality on Social Platforms.* ASONAM 2025.
