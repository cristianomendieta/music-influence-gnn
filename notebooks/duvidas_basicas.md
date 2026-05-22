# Dúvidas básicas sobre a pesquisa

Documento didático em linguagem simples para fixar conceitos-chave antes da
apresentação da Phase 1. Vai sendo atualizado conforme novas dúvidas aparecem.

---

## 1. Que problema essa pesquisa resolve?

### Resposta curta

A gente quer **prever a curva de popularidade de uma música no Spotify ao
longo do tempo** — e mostrar que olhar para a *rede* (artistas, gêneros,
músicas vizinhas no chart) prevê melhor do que olhar para cada música
isoladamente.

### Em mais detalhe

Não é simplesmente "essa música vai bombar, sim ou não?". O que a gente
prevê é uma **série temporal**: para cada música, quanto ela vai pesar no
chart em cada semana.

Tecnicamente o alvo é o **`rank_score`** — um número entre 0 e 0,5 que
resume a posição da música no Top 200 ou no Viral 50 daquela semana
(quanto mais perto do topo, maior o score; suavizado por média móvel de
7 dias).

A gente faz isso em **dois regimes** de avaliação:

| Regime | O que entra | O que sai | Para que serve |
|---|---|---|---|
| **Fit retroativo** | Toda a curva real da música | A curva ajustada | Comparar 1-pra-1 com o SIR do Oliveira (BraSNAM 2025) |
| **Predição genuína** | Só dados até a semana *t* | Score em *t + k* (k = 1, 7, 14, 30 dias) | Mostrar que o GNN serve para projetar o futuro, não só para descrever o passado |

### Exemplo concreto

Pega a música **"Batom de Cereja - Ao Vivo" (Israel & Rodolffo)**, uma
das figuras-chave do paper original — começou viral e virou hit longo.

- **Entrada do modelo (predição genuína em t = 2021-04-15):**
  - Histórico de `rank_score` semanal da música nas últimas 8 semanas.
  - O grafo da semana 2021-04-15: artista Israel & Rodolffo, gênero
    *sertanejo*, vizinhança de músicas que estavam no chart junto.
  - Features acústicas (energy, danceability, valence, etc.) e
    metadados (popularity, explicit, total_streams).

- **Saída do modelo:**
  - `rank_score` previsto para 2021-04-16 (k=1), 2021-04-22 (k=7),
    2021-04-29 (k=14) e 2021-05-15 (k=30).

- **Como avaliamos:**
  - Comparamos o número previsto com o `rank_score` real daqueles dias.
  - RMSE da nossa previsão *vs.* RMSE do SIR clássico *vs.* RMSE do
    wave-based (Oliveira ASONAM 2025) — todos rodando no mesmo regime.
  - Hipótese: o GNN ganha justamente em casos como esse (hit longo
    com re-emergência) porque usa o sinal estrutural — sabe que a
    música tem um artista de sertanejo ativo e que vizinhos do chart
    sustentam a popularidade.

### Por que isso é uma pergunta nova

Toda a linha do Oliveira et al. (2025 BraSNAM, 2025 ASONAM, 2025 IEEE
Access) trata cada música **isoladamente**, com modelos populacionais
(SIR, wave-based). Nenhum desses trabalhos usa o fato de que músicas
compartilham artistas, gêneros e momento de chart. A nossa tese é:
**esse sinal relacional explica variância que o modelo populacional
deixa na mesa** — especialmente nos hits longos, onde o SIR tem RMSE
2× maior que nos virais.

---

## 2. Como é o grafo que vamos usar?

### Resposta curta

É um **grafo heterogêneo** — quer dizer, tem **vários tipos de nó** e
**vários tipos de aresta** convivendo na mesma estrutura. Três tipos de
nó (música, artista, gênero) e quatro tipos de aresta ligando eles.

Esse grafo está construído e salvo em
[data/processed/graph/hetero_full.pt](../data/processed/graph/hetero_full.pt)
no formato `HeteroData` do PyTorch Geometric.

### Nós (os "círculos" do grafo)

| Tipo de nó | Quantos | O que cada nó representa | Features iniciais |
|---|---|---|---|
| **música** | 6.526 | uma faixa do Spotify que apareceu no Top 200 ou no Viral 50 BR entre 2017 e 2022 | 9 features acústicas (danceability, energy, valence, …) + popularity + explicit + song_type + total_streams + dias_no_chart |
| **artista** | 1.701 | um artista brasileiro com música no chart | num_hits, num_collab_hits, anos no chart, número de gêneros |
| **gênero** | 530 | um gênero musical (ex.: *sertanejo*, *funk carioca*, *forró*, *pagode*) | embedding aprendido ou one-hot |

*Obs.:* o spec inicial estimava 6.469 músicas; o universo real ficou em
6.526 (top200 ∪ viral50-com-features-acústicas). A tolerância foi
ajustada para ±100.

### Arestas (as "linhas" do grafo)

| Aresta | Direção | De onde vem | O que ela diz |
|---|---|---|---|
| `artista → música` (`performs`) | direcionada | listas `artist_ids` na tabela de músicas | "esse artista interpreta essa música" (com papel: principal ou feat) |
| `artista — gênero` (`has_genre`) | não-direcionada | listas `genres` na tabela de artistas | "esse artista atua nesse gênero" |
| `música → música` (`cotrajectory`) | direcionada (quem entrou antes aponta para quem entrou depois) | charts diários | "essas duas músicas ficaram ≥7 dias juntas no Top 200/Viral 50" |
| `gênero — gênero` (`cooccurs`) | não-direcionada | rede de gêneros do MGD+ | "esses dois gêneros costumam aparecer juntos no chart" (com peso) |

### Como visualizar

Pense numa rede social, mas em vez de "pessoa segue pessoa", você tem:

```
                    ┌─────────┐
                    │ gênero  │──┐  cooccurs
                    │sertanejo│  │
                    └────┬────┘  ▼
                         │     ┌─────────┐
                  has_genre    │ gênero  │
                         │     │  forró  │
                         ▼     └────┬────┘
                    ┌─────────┐    │
                    │ artista │◄───┘ has_genre
                    │Israel & │
                    │Rodolffo │
                    └────┬────┘
                  performs│
                         ▼
                    ┌─────────┐ cotrajectory  ┌─────────┐
                    │ música  │──────────────►│ música  │
                    │ "Batom  │               │"Evidênc.│
                    │ Cereja" │◄──────────────│  Ao V." │
                    └─────────┘ cotrajectory  └─────────┘
```

Cada nó carrega o seu vetor de features. Cada aresta carrega o tipo da
relação (e, em alguns casos, atributos extras — peso de co-ocorrência,
número de dias juntos no chart, distância média de posição).

### Por que heterogêneo (em vez de um grafo "comum")?

Num grafo comum (homogêneo), todo nó é da mesma natureza e toda aresta
significa a mesma coisa. Aqui isso seria perda de informação: uma aresta
*artista → música* não significa o mesmo que uma aresta *música → música*.
O `HeteroGraphSAGE` que vamos usar na Phase 2 aprende **um conjunto
diferente de pesos para cada tipo de aresta**, então o modelo consegue
distinguir "co-trajetória no chart" de "participação do artista".

### E a parte temporal?

O grafo `hetero_full.pt` é a **estrutura completa** (todas as músicas,
artistas, gêneros, todas as arestas que existiram em algum momento).
Sobre ele, cada nó/aresta carrega um campo `first_seen_week`.

Na Phase 2, a gente vai gerar **snapshots semanais**: para a semana *t*,
mantém só nós e arestas com `first_seen_week ≤ t` (máscara `mask_until`).
Cada snapshot vira um embedding por música; uma GRU lê a sequência de
embeddings (janela de W semanas) e gera a previsão final.

---

## Próximas dúvidas a documentar

- [ ] Como o `rank_score` é calculado, passo a passo, a partir dos charts?
- [ ] Diferença prática entre "viral" e "hit" no nosso recorte.
- [ ] Por que o threshold de co-trajetória é 7 dias?
- [ ] O que é exatamente "wave-based" do Oliveira ASONAM 2025?
- [ ] Como funciona o GRU sobre os embeddings semanais?
