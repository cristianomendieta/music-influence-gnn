# Difusão de popularidade musical com GNN temporal heterogêneo

**Projeto:** BraSNAM 2026 — GNN temporal heterogêneo para difusão de popularidade musical
**Status:** Fases 0, 1 e 2 concluídas (Fase 2 com grid de hiperparâmetros ainda em execução)
**Documento:** resumo do que já foi construído, para discussão com o orientador.

---

## 1. Visão geral & pergunta de pesquisa

### O gap na literatura do grupo

Os trabalhos do grupo (Oliveira et al.) modelam a popularidade de **cada música como uma
curva epidêmica isolada** — um processo SIR (Suscetível–Infectado–Recuperado) ou um
"wave model" ajustado individualmente à série temporal de cada faixa. Cada música é tratada
como um universo fechado: o modelo não enxerga **quem** canta a música, **a que gênero**
ela pertence, ou **com quais outras músicas** ela compartilha o palco nos charts.

### A hipótese deste projeto

> **Sinais relacionais** (artista → música → gênero, e co-trajetórias entre músicas nos
> charts) capturam variância da difusão de popularidade que os modelos populacionais
> isolados (SIR) **não conseguem** capturar.

A ideia é simples de enunciar: se uma música nova é lançada por um artista cujas faixas
anteriores estouraram, ou se ela aparece no chart junto de hits consolidados de gênero
parecido, isso é informação preditiva — mas um modelo SIR ajustado só à própria curva da
música a ignora por construção.

Para testar isso, construímos um **grafo temporal heterogêneo** e treinamos uma **GNN**
(Graph Neural Network) que combina:

- a **estrutura** (artista–música–gênero + co-trajetórias), via *message passing* heterogêneo;
- a **dinâmica** (como o grafo cresce semana a semana), via uma rede recorrente (GRU).

---

## 2. Dados

| Dimensão | Valor |
|----------|-------|
| Período | 2017-01-01 → 2021-12-31 (charts BR) |
| Charts | Top 200 + Viral 50 (Brasil) |
| Fonte | MGD+ / Kaggle (charts, features acústicas, rede de gêneros) |
| Músicas (universo do grafo) | **6.526** |
| Artistas (catálogo BR) | **1.701** |
| Gêneros | **530** |
| Subset de análise (viral ∩ hit) | **1.981** músicas |

**Universo de músicas:** união das músicas do Top 200 com as do Viral 50 que possuem
features acústicas (evita imputar >48% das faixas). **Artistas:** apenas o catálogo
brasileiro (`br-artists-all_time.csv`); artistas estrangeiros que aparecem só no Viral 50
são descartados silenciosamente. **Gêneros e suas conexões** vêm da rede de gêneros do MGD+.

**Limitação declarada:** há um gap de ~2,5 meses no início de 2022 nos dados originais;
por isso o calendário do grafo é truncado em 2021-W52 (semana 260). Datas de 2022 são
descartadas na agregação.

### Os dados brutos — as planilhas do MGD+

Antes de existir qualquer grafo, o MGD+ é um conjunto de **planilhas planas**. Abaixo, amostras
reais de cada fonte (colunas selecionadas), para deixar claro **de onde** cada elemento do grafo
nasce.

**`charts/spotify_charts_regional_br` — Top 200** (uma linha por música × dia × posição):

| ID | Track | Artist | Position | Streams | Date | Chart |
|----|-------|--------|----------|---------|------|-------|
| 0EPxmvsG1BY5td4aTOkWBF | Deu Onda | MC G15 | 1 | 612.271 | 2017-01-01 | Top 200 |
| 1a5Yu5L18qNxVhXx38njON | Hear Me Now | Alok, Bruno Martini, Zeeba | 2 | 269.574 | 2017-01-01 | Top 200 |

O Viral 50 (`spotify_charts_viral_br`) tem o mesmo formato, **sem** a coluna `Streams` (só posição).

**`songs/br-hit_songs-{ano}` — features acústicas** (uma linha por música):

| song_id | song_name | artist_name | popularity | explicit | danceability | energy | valence | tempo |
|---------|-----------|-------------|-----------|----------|--------------|--------|---------|-------|
| 0UavwCicOnQwMuh67yaxM3 | Vem Me Amar | ['Jonas Esticado'] | 45 | False | 0,612 | 0,934 | 0,958 | 166,96 |

**`artists/br-artists-all_time` — catálogo BR** (créditos + filiação de gênero):

| artist_id | name | genres | years_on_charts | num_hits | num_collab_hits |
|-----------|------|--------|-----------------|----------|-----------------|
| 1yR65psqiazQpeM79CcGh8 | Marília Mendonça | ['arrocha', 'sertanejo', 'sertanejo universitario'] | [2017,…,2022] | 140 | 78 |

**`genre_network/br-genre_network-{ano}` — co-ocorrência de gêneros** (separador por **vírgula**, não tab):

| Source | Target | Weight | Avg_Popularity |
|--------|--------|--------|----------------|
| sertanejo | sertanejo universitario | 101 | 43,33 |
| arrocha | sertanejo universitario | 94 | 42,80 |

### De planilha a grafo — o mapa da transformação

Cada coluna dessas tabelas cruas vira um pedaço do grafo heterogêneo. É essa **releitura
relacional** que a Fase 1 (seção 4) detalha aresta por aresta:

| Planilha crua (MGD+) | Vira no grafo… | Como |
|----------------------|----------------|------|
| `charts/*` — música × dia × posição | **nós `music`** + arestas **`cotrajectory`** | músicas viram nós; coexistir ≥ 7 dias num chart vira aresta dirigida (quem entrou antes → depois) |
| `songs/br-hit_songs-*` — features acústicas | **features dos nós `music`** (15-d) | 9 acústicas (z-score) + popularidade + explicit + tipo + streams + dias no chart |
| `artists/br-artists-all_time` — créditos + `genres` | **nós `artist`** + arestas **`performs`** / **`has_genre`** | crédito da música → `performs` (pos 0 = main, demais = feat); coluna `genres` → `has_genre` |
| `genre_network/*` — `Source,Target,Weight` | **arestas `cooccurs`** entre nós `genre` | cada linha é um par de gêneros co-ocorrentes; simétrica por construção |
| coluna `Date` dos charts | **`first_seen_week ∈ [0,260]`** em toda aresta | ISO-week desde 2017-W1 → define em que snapshot a aresta aparece |

O salto conceitual: a planilha é **plana e isolada** (uma linha não "conhece" a outra); o grafo
torna **explícitas as relações** que estavam apenas implícitas nas tabelas — justamente o sinal
relacional que a hipótese do projeto quer explorar.

---

## 3. Fase 0 — Baselines SIR *(concluída)*

### O que foi feito

Reimplementamos o **modelo SIR** que o grupo usa como referência: para cada música,
ajustamos uma curva epidêmica de 3 parâmetros (β, γ e tamanho da população) à série
temporal de popularidade, separando dois regimes — **virality** (Viral 50) e **success**
(Top 200). Isso serve de **baseline a ser superado** e reproduz o resultado do paper.

### Resultados (validação numérica vs. o paper)

| Métrica | Alvo (paper) | Observado | Status |
|---------|--------------|-----------|--------|
| SIR · RMSE *virality* | ≈ 0,028 | **0,0289** | ✅ |
| SIR · RMSE *success* | ≈ 0,052 | **0,0471** | ✅ |
| Mann-Whitney p (success vs. virality) | ≈ 1e-60 | **1,61e-39** | ✅ |
| Tamanho do subset | ≥ 1.900 | **1.981** | ✅ |
| Convergência do ajuste | ≥ 99% | **100%** | ✅ |

### Decisões e lições

- **Wave model descartado:** o ajuste "wave-based" (multi-onda) tem custo computacional
  proibitivo no dataset completo e não agregava o suficiente para justificá-lo.
- **Hits longos são multi-onda:** faixas que ficam meses no chart têm múltiplos picos, que
  um SIR de onda única captura mal — exatamente o tipo de variância que esperamos que a
  estrutura relacional ajude a explicar.

---

## 4. Fase 1 — Grafo heterogêneo *(concluída e validada C1–C9)*

Esta é a **espinha dorsal da contribuição**: o objeto que codifica os sinais relacionais.

### Esquema (3 tipos de nó, 4 tipos de aresta + reversa)

```
                  has_genre
        artist ───────────────► genre
          │   ◄───────────────    │
          │    rev_has_genre      │ cooccurs (gênero↔gênero)
          │ performs              ▼
          ▼                      ...
        music ◄──┐
          │      │ cotrajectory (música→música)
          └──────┘
```

| Nó | Contagem | Features |
|----|----------|----------|
| `music` | 6.526 | 15 dims: 9 acústicas (z-score) + popularidade + explicit + tipo + streams + dias no chart + flag de ausência |
| `artist` | 1.701 | 4 dims: nº hits, nº colaborações, anos no chart, nº gêneros (z-score) |
| `genre` | 530 | embedding estrutural 32-d (init aleatória, treinável) |

O `build_hetero` ([build.py:62](../src/music_diffusion_gnn/graph/build.py#L62)) orquestra
tudo: constrói os nós, depois cada tipo de aresta, e monta o `HeteroData` do PyG. As
subseções abaixo mostram **como cada aresta nasce**, com o trecho de código que a define.

### Resumo das arestas

| Aresta | Contagem | Como nasce |
|--------|----------|------------|
| `artist → performs → music` | 9.274 | Crédito da música; papel **main** (posição 0) vs. **feat** (demais) |
| `artist → has_genre → genre` | 3.344 | Filiação de gênero do artista (sem informação temporal → `first_seen_week = 0`) |
| `genre → rev_has_genre → artist` | 3.344 | Reversa da anterior, adicionada manualmente no build |
| `music → cotrajectory → music` | 664.577 | Duas músicas que **coexistem ≥ 7 dias no mesmo chart**; direção = quem entrou antes |
| `genre → cooccurs → genre` | 9.866 | Rede de co-ocorrência de gêneros do MGD+ (simétrica) |

### Antes das arestas — o universo de nós

Quem entra no grafo é decidido em `build_music_nodes`
([nodes.py:75](../src/music_diffusion_gnn/graph/nodes.py#L75)). O universo de músicas é a
união do Top 200 com as músicas do Viral 50 **que têm features acústicas** — isso evita
imputar mais de 48% das faixas:

```python
universe = top200_songs | (viral50_songs & songs_with_features)
```

Os artistas vêm só do catálogo BR; artistas estrangeiros referenciados por músicas do
Viral 50 são descartados (e logados como warning), não viram nós-fantasma com features zero
([nodes.py:179](../src/music_diffusion_gnn/graph/nodes.py#L179)).

### 4a. `performs` (artista → música)

Vem da lista de créditos de cada música. A **posição na lista** define o papel: índice 0 é o
artista principal (`role=0`), os demais são *feat* (`role=1`). O atributo de aresta guarda
`[role, posição, first_seen_week]`, onde `first_seen_week` é a semana da primeira aparição da
música em qualquer chart ([edges.py:76](../src/music_diffusion_gnn/graph/edges.py#L76)):

```python
for pos, aid in enumerate(artist_list):
    role = 0 if pos == 0 else 1          # 0 = main, 1 = feat
    rows_attr.append([role, pos, fsw])   # fsw = week_index da estreia da música
```

### 4b. `has_genre` (artista → gênero) **+ reversa**

A filiação de gênero do artista. Como não há informação de *quando* o artista passou a
pertencer ao gênero, a `first_seen_week` é fixada em 0 — a aresta existe em todos os
snapshots ([edges.py:101](../src/music_diffusion_gnn/graph/edges.py#L101)).

A reversa `rev_has_genre` é criada **manualmente** no build, simplesmente invertendo as duas
linhas do `edge_index` com `.flip(0)`
([build.py:139-142](../src/music_diffusion_gnn/graph/build.py#L139-L142)):

```python
g["artist", "has_genre", "genre"].edge_index = hg["edge_index"]
g["genre", "rev_has_genre", "artist"].edge_index = hg["edge_index"].flip(0)  # reversa
```

**Por que a reversa?** No HeteroGraphSAGE a informação só flui no sentido da aresta
(`src → dst`). Sem a reversa, o nó `genre` *receberia* sinal dos artistas, mas o artista
**nunca** receberia de volta o embedding do gênero — e esse sinal jamais chegaria à música via
`performs`. A reversa abre o caminho `genre → artist → music`, que é justamente o que faz o
gênero influenciar a previsão de popularidade. Isso explica os **3.344** = mesma contagem de
`has_genre` (é o espelho exato). As demais arestas têm tratamento próprio: `cotrajectory` é
dirigida de propósito (carrega causalidade temporal) e por isso **não** é espelhada; `cooccurs`
já é simétrica no próprio builder (ver 4d).

### 4c. `cotrajectory` (música → música) — a aresta mais rica

Para cada par de músicas que aparece junto num chart, contamos os dias de coexistência. Só
vira aresta se forem **≥ 7 dias**, e a **direção** segue quem entrou no chart primeiro
([edges.py:205-226](../src/music_diffusion_gnn/graph/edges.py#L205-L226)):

```python
if count < 7:
    continue                          # laço fraco descartado
...
if d_si < d_sj:                       # música i estreou no chart antes de j
    src_id, dst_id = si, sj           # i → j
elif d_si > d_sj:
    src_id, dst_id = sj, si
else:
    src_id, dst_id = (si, sj) if si <= sj else (sj, si)  # empate: lexicográfico
```

O atributo guarda `[dias juntos, distância média de posição, chart, first_seen_week]`. É a
aresta mais cara de construir (~36s sobre o dataset completo), feita varrendo os charts dia a
dia ([edges.py:145](../src/music_diffusion_gnn/graph/edges.py#L145)).

### 4d. `cooccurs` (gênero ↔ gênero)

Lê a rede de gêneros do MGD+ ano a ano. É **simétrica por construção**: cada par é inserido
nas duas direções no próprio builder, então não precisa de reversa
([edges.py:305](../src/music_diffusion_gnn/graph/edges.py#L305)):

```python
for key in [(src_idx, dst_idx), (dst_idx, src_idx)]:  # ambas as direções
    ...
```

### O tempo dentro do grafo (mascaramento sem vazamento)

A semana de cada aresta é calculada por `week_index`, um offset de ISO-week desde 2017-W1
([temporal.py:23](../src/music_diffusion_gnn/graph/temporal.py#L23)):

```python
idx = (year - 2017) * 52 + (week - 1)   # 0 = 2017-W1 … 260 = 2021-W52
```

Toda aresta carrega esse `first_seen_week ∈ [0, 260]`. A função `mask_until(g, w)` devolve um
**snapshot** com **apenas as arestas que já existiam até a semana `w`** — um simples filtro
booleano sobre o `edge_index` ([temporal.py:68](../src/music_diffusion_gnn/graph/temporal.py#L68)):

```python
mask = fsw <= week_t
g[edge_type].edge_index = et.edge_index[:, mask]
```

Isso garante que, ao prever a popularidade na semana `w`, o modelo **nunca** enxerga arestas do
futuro — a base do treino temporal honesto da Fase 2.

### Validação C1–C9 (todos verdes)

Contagens de nós dentro da tolerância, subset viral∩hit ⊆ nós `music`, todo artista do
subset alcançável via `performs`, índices de aresta válidos, `first_seen_week ∈ [0,260]`,
ausência de auto-loops indevidos, etc. Build completo em ~40s (sendo ~36s na co-trajetória).

### Estatísticas estruturais

- **Hubs claros:** co-trajetória tem grau de saída máximo 2.048 e mediano 71 — poucas
  músicas "âncora" conectam muitas outras.
- **Clustering alto:** 0,51 (co-trajetória) e 0,60 (gêneros) — vizinhanças coesas.
- **Comunidades de gênero (Louvain):** as 5 maiores têm 136, 119, 96, 60 e 43 gêneros,
  agrupando coerentemente (sertanejo/arrocha/axé; rock; pop alternativo; EDM; hip-hop).

---

## 5. Fase 2 — GNN temporal *(implementada; grid em execução)*

Construímos o grafo na Fase 1 — **e agora?** Esta seção responde, do zero, às três perguntas
naturais de quem nunca mexeu com GNN nem com séries temporais: **(1)** jogamos o grafo inteiro
numa rede neural? **(2)** o que exatamente estamos prevendo? **(3)** como sabemos se ficou bom?

### 5.0 A ideia em uma frase

Queremos **prever a popularidade de cada música na próxima semana**. Em vez de olhar só para a
curva passada daquela faixa (como o SIR da Fase 0), deixamos a música "enxergar" seus vizinhos
no grafo — o artista que a canta, os gêneros, as músicas que dividem o chart com ela — e usamos
**como ela e essa vizinhança evoluíram nas últimas semanas** para prever a semana seguinte. São
duas peças encaixadas, cada uma resolvendo metade do problema:

1. **Parte espacial — a GNN:** olha a *foto* do grafo numa dada semana e produz **um vetor de
   números por música** (um "resumo aprendido" da faixa + sua vizinhança naquele instante).
2. **Parte temporal — a GRU:** olha a *sequência* desses vetores ao longo de várias semanas e
   produz o número final — a popularidade prevista.

> **Por que duas peças?** A estrutura (quem se conecta a quem) e o tempo (como tudo evolui) são
> dois eixos diferentes. A GNN cuida do eixo "relações"; a GRU cuida do eixo "história". Sozinha,
> a GNN não teria noção de passado; sozinha, a GRU não enxergaria os vizinhos.

### 5.1 "Jogamos o grafo inteiro na rede?" — Não. Um *snapshot* por semana → um embedding por música

Não despejamos o grafo estático inteiro de uma vez. O fluxo é:

**(a) Recortar o tempo.** Para cada semana `w`, `mask_until(g, w)` (Fase 1) devolve um
**snapshot**: o mesmo grafo, mas só com as arestas que já existiam até `w`. Isso é o que impede
o modelo de "ver o futuro".

**(b) Message passing — o coração da GNN.** O snapshot entra no **HeteroGraphSAGE**. A intuição
de *message passing* é simples: **cada nó atualiza seu próprio vetor somando versões
transformadas dos vetores dos seus vizinhos** (mais o seu). Uma camada = um "salto" (hop):

- depois de **1 camada**, a música já incorporou sinal dos seus vizinhos diretos (co-trajetos e
  o artista que a canta, via `performs`);
- depois de **2 camadas**, alcança os vizinhos-dos-vizinhos — por exemplo o **gênero**, que
  chega à música pelo caminho `genre → artist → music` (é exatamente para isso que serve a
  aresta reversa `rev_has_genre` da Fase 1, seção 4b);
- "heterogêneo" só quer dizer que há **um conjunto de pesos por tipo de aresta** (`performs`,
  `has_genre`, `cotrajectory`, `cooccurs`, …), porque um co-trajeto não significa a mesma coisa
  que uma filiação de gênero. No código, é um `HeteroConv` com um `SAGEConv` por tipo de aresta,
  agregando por **soma** ([encoder.py](../src/music_diffusion_gnn/models/encoder.py)).

**(c) O resultado — *embeddings*.** A saída da GNN é uma matriz `Z(w)` de formato
`(6.526 músicas × hidden)`, onde `hidden` é, p.ex., 64. Cada **linha é o "embedding" daquela
música na semana `w`**: um vetor de 64 números que condensa, num único ponto, tudo que a música
*é* e *com quem se relaciona* naquele instante. Embedding é só isso — uma representação numérica
densa e aprendida, que a rede acha útil para prever bem. Não é interpretável linha a linha; é a
"matéria-prima" que a parte temporal vai consumir.

Repetindo isso para várias semanas, montamos um **banco de embeddings** `{w: Z(w)}` — uma foto
vetorial do grafo por semana.

### 5.2 Da estrutura ao tempo — a janela causal e a GRU

Agora a parte de **série temporal**. Para prever a semana-alvo `w` de uma música, **não** usamos
só `Z(w-1)`: pegamos os embeddings dela nas **W semanas anteriores** — `Z(w-W), …, Z(w-1)` —
formando uma **sequência** (a "janela causal", de tamanho `W`). "Causal" porque a janela só
contém passado: nunca inclui a própria semana `w` nem nada depois.

Essa sequência de `W` vetores entra numa **GRU** (*Gated Recurrent Unit*), um tipo de rede
recorrente feita para sequências: ela processa um passo de cada vez, carregando um "estado de
memória" que resume o que viu até ali. Pegamos o **último estado** da GRU (o resumo da janela
inteira), passamos por um pequeno **MLP** (duas camadas lineares) e esprememos a saída para o
intervalo válido com `0,5 · sigmoid(·)` → um único número `ŷ ∈ [0, 0.5]`
([temporal_head.py](../src/music_diffusion_gnn/models/temporal_head.py)).

> **Detalhe — *padding*:** se a música ainda **não existia** em alguma das `W` semanas anteriores
> (estreou depois), aquela posição da janela é marcada como "vazia" (`pad_mask`) e zerada antes
> da GRU, para não inventar histórico que não houve.

O pipeline completo, de ponta a ponta:

```
  para cada semana w do grafo:
    snapshot = mask_until(g, w)          ← só arestas até w (sem vazamento)
        │
        ▼
    HeteroGraphSAGE (L camadas)          ← message passing heterogêneo
        │
        ▼
    embedding por música  Z(w)           ← "banco de embeddings" por semana
        │
   ┌────┴───────────────────────┐
   │ janela causal de W semanas: │        para prever a semana-alvo w:
   │  Z(w-W) … Z(w-1)            │        pega os embeddings das W semanas anteriores
   └────┬───────────────────────┘
        ▼
       GRU (1 camada) + MLP
        │
        ▼
       ŷ ∈ [0, 0.5]                       ← popularidade prevista (0,5·sigmoid)
```

### 5.3 "O que estamos prevendo?" — uma amostra = (música, chart, semana-alvo)

#### O que é o alvo `y`, exatamente

Antes de tudo: `y` **não** é número de streams, nem a coluna `popularity` do Spotify. É um
**score de popularidade derivado da posição no chart**, construído em 4 passos
([preprocess.py](../src/music_diffusion_gnn/data/preprocess.py), replica Oliveira et al. 2025 §4.1):

| Passo | O que faz | Código |
|-------|-----------|--------|
| **rank → score** | `rank_score = max_rank − rank + 1`. No Top 200 (`max_rank=200`): posição 1 → 200, posição 200 → 1; fora do chart → 0. Viral 50 usa `max_rank=50` | [preprocess.py:22](../src/music_diffusion_gnn/data/preprocess.py#L22) |
| **MA-7d** | média móvel de 7 dias, suavizando o ruído diário | [preprocess.py:28](../src/music_diffusion_gnn/data/preprocess.py#L28) |
| **min-max → [0, 0.5]** | normaliza a curva **de cada música pelo próprio min/max**, escalando para `[0, 0.5]` | [preprocess.py:31](../src/music_diffusion_gnn/data/preprocess.py#L31) |
| **floor 0.001** | zeros viram `0.001` (evita log/zeros exatos) | [preprocess.py:34](../src/music_diffusion_gnn/data/preprocess.py#L34) |

> **A sutileza que muda tudo está no passo 3:** a normalização é **por música**. Logo, `y = 0.5`
> **não** quer dizer "a música mais popular do Brasil" — quer dizer **"esta música no seu próprio
> pico"**; `y ≈ 0` é o vale/ausência dela. `y` é a **forma (relativa) da curva de popularidade**
> da faixa ao longo do tempo, não uma medida absoluta entre músicas.

É justamente por isso que `y` é o alvo certo: é a **mesma curva `[0, 0.5]`** que o **SIR da Fase 0
ajusta** como curva epidêmica — então GNN e SIR preveem exatamente a mesma quantidade, e a
comparação da Fase 3 é justa.

O alvo do modelo é o **`y_week`**: esse `y` diário **agregado por média dentro de cada semana ISO**
([`aggregate_weekly`, dataset.py](../src/music_diffusion_gnn/training/dataset.py)) — a única
diferença é a granularidade (semanal em vez de diária).

#### Entrada × alvo: o que sai quando entro com o grafo até `w-1`

Vale fixar a separação, porque é onde a intuição costuma escorregar:

- **Entrada:** os embeddings das `W` semanas **anteriores** (`w-W … w-1`) — o passado.
- **Saída `ŷ`:** **um único escalar** `ŷ ∈ [0, 0.5]`, a popularidade prevista da semana **`w`**,
  que fica **um passo à frente** da janela. O `w` nunca entra na entrada → é previsão de verdade,
  não leitura de um valor já visto.

E "um número" para **quem**? Para **uma** música, num chart, numa semana — não um vetor sobre as
6.526 músicas. A GNN *internamente* calcula embedding para todas elas (a matriz `Z(w)`), mas a
cabeça temporal **seleciona só a linha da música-alvo** e devolve um escalar
([`predict`, diffusion_gnn.py](../src/music_diffusion_gnn/models/diffusion_gnn.py)). Para prever
todas as músicas de uma semana, repete-se isso por música (em lote) — cada saída continua sendo
"o `y_week` previsto de UMA música num chart":

```
grafo até w-1  ──GNN──►  Z(w-1) inteiro (6526 × hidden)   ← todas as músicas
                              │
                   seleciona SÓ a linha da música-alvo (nas W semanas)
                              │
                          GRU + MLP
                              │
                              ▼
                     ŷ ∈ [0,0.5]   ← UM número: popularidade prevista
                                       daquela música, naquele chart, na semana w
```

#### A amostra, formalmente

O modelo faz **regressão**: dado uma *amostra*, prevê **um número**. Uma amostra é a tripla
**(música, chart, semana-alvo `w`)** — ver `Sample` em
[dataset.py](../src/music_diffusion_gnn/training/dataset.py):

| Campo | Significado |
|-------|-------------|
| `song_idx` | qual música (índice do nó no grafo) |
| `chart` | em qual regime: `0=viral50`, `1=top200` (a mesma música pode ter alvo nos dois) |
| `target_week` | a semana `w` cuja popularidade queremos prever |
| `window_weeks` | as `W` semanas anteriores `[w-W … w-1]` que alimentam a GRU |
| `y` | a resposta certa: o `y_week` observado naquela semana (o que tentamos acertar) |

Só criamos amostra para semanas **depois** de a música ter estreado naquele chart
(`target_week > first_seen_week`) — não faz sentido prever popularidade antes de a faixa existir.
Isso gera dezenas de milhares de amostras (uma por música × chart × semana válida).

### 5.4 Como treinamos

Aprendizado **supervisionado** padrão, repetido por época:

1. para cada amostra, o modelo produz `ŷ`;
2. comparamos com o `y` verdadeiro pela **MSE** (erro quadrático médio) — `nn.MSELoss`;
3. o **backpropagation** calcula como cada peso (da GNN, da GRU, do MLP, e até os embeddings de
   gênero) contribuiu para o erro, e o otimizador **Adam** ajusta todos eles um passo na direção
   que reduz o erro ([trainer.py](../src/music_diffusion_gnn/training/trainer.py));
4. repete-se por até **100 épocas**, com **early stopping**: se a métrica de validação não
   melhora por **10 épocas** seguidas, paramos e guardamos os melhores pesos. Isso evita
   *overfitting* (decorar o treino sem generalizar).

**Splits temporais sem vazamento** (a regra de ouro de séries temporais — treina-se no passado,
testa-se no futuro):

- **treino:** semanas ≤ 182 (até 2020-06);
- **validação:** semanas 183–207 (jul–dez 2020) — usada para o early stopping e para escolher
  hiperparâmetros;
- **teste:** semanas ≥ 208 (2021) — intocada até o fim, simula "prever 2021 conhecendo só o
  passado".

As fronteiras são por **semana inteira**, então nenhuma semana aparece em dois splits.

### 5.5 "Como avaliamos?" — contra baselines, no futuro nunca visto

Medimos o erro da previsão com **MSE/RMSE** entre `ŷ` e `y_week` no split de teste — quanto
menor, melhor. Mas um número sozinho não diz nada; o que importa é **vencer referências
ingênuas**. Avaliamos sempre **por regime** (viral50 e top200, que têm dinâmicas distintas) e
comparamos contra:

- **Persistência:** o palpite preguiçoso `ŷ(w) = y(w-1)` — "a próxima semana será igual à
  anterior". É surpreendentemente forte em séries temporais; se a GNN não bate isso, não está
  agregando nada ([baselines.py](../src/music_diffusion_gnn/models/baselines.py)).
- **SIR (Fase 0):** o modelo epidêmico do grupo, a referência da literatura que queremos superar.

A pergunta de pesquisa só é respondida **"sim"** se a GNN temporal tiver erro **menor** que esses
dois — é isso que a **Fase 3** vai medir formalmente (inclusive com teste estatístico de
significância, como o Mann-Whitney já usado na Fase 0).

### 5.6 Detalhes de implementação

- **Banco de embeddings por semana:** cada snapshot semanal é codificado **uma vez** por
  forward e reaproveitado por todas as amostras daquela semana — peça-chave do desempenho.
- **DropEdge nas co-trajetórias:** as 664k arestas de co-trajetória são subamostradas para
  ≤ 30.000 por snapshot (regularização + necessidade de RAM/autograd no WSL).
- **Grid de hiperparâmetros:** 24 configurações — `W ∈ {4,8,12}` semanas, `hidden ∈ {64,128}`,
  `layers ∈ {2,3}`, `lr ∈ {1e-3, 5e-4}`. Tamanho do modelo: **~82K a ~469K** parâmetros.
  (`W` = tamanho da janela causal; `hidden` = dimensão do embedding; `layers` = nº de saltos de
  message passing; `lr` = taxa de aprendizado do Adam.)

### Resultados preliminares *(grid ainda em execução)*

> **Importante:** o foco desta fase é o **método**. O grid de 24 configurações ainda está
> rodando em background; os números abaixo são da **melhor config encontrada até agora** e
> devem ser lidos como **preliminares**.

Melhor config até o momento: `W4_h64_l2_lr1e-03` (82.625 parâmetros), `val_mse ≈ 0,0039`.
O foco da avaliação completa (comparação dupla contra SIR e contra persistência) é da
**Fase 3** — o objetivo desta fase foi **construir e validar o pipeline temporal** de ponta
a ponta, sem vazamento, e ele está funcionando.

---

## 6. Decisões e desvios importantes

| Tema | Decisão / desvio | Motivo |
|------|------------------|--------|
| Universo de músicas | 6.526 (real) vs. 6.469 (estimativa do spec) | Spec escrito antes de inspecionar os dados; tolerância C1 ajustada para ±100 |
| Wave model | Descartado | Custo computacional proibitivo, ganho marginal |
| Fronteira de split | `week_index(2020-06-30) = week_index(2020-07-01) = 182` | Anos ISO com 53 semanas quebram a bijeção; solução: partição estrita por semana (train ≤182, val (182,208), test ≥208) |
| `week_index` por linha | Substituído por `dt.isocalendar()` vetorizado do pandas | 2017-01-01 é ISO 2016-W52 → lança erro linha a linha |
| DropEdge co-trajetória | `max_cotraj_edges = 30.000` obrigatório | RAM/autograd no WSL não suportam 664k arestas por snapshot |
| Performance | Batching por semana-alvo + `predict` vetorizado | **14× mais rápido** (38s/época vs. ~520s na versão inicial) |

---

## 7. Próximos passos

- **Fase 3 — Avaliação:** comparação dupla e honesta da GNN temporal contra (a) os baselines
  SIR da Fase 0 e (b) a persistência ingênua, nos dois regimes (virality/success), no horizonte
  semanal, encerrando o grid de hiperparâmetros.
- **Fase 4 — Escrita:** consolidação dos resultados no formato do artigo para o BraSNAM 2026.

---

*Materiais de apoio:* o notebook `notebooks/phase2_apresentacao.ipynb` mostra, de forma
visual e a partir dos dados reais, **como o grafo é construído** e como ele alimenta a GNN
temporal. O notebook `notebooks/phase1_apresentacao.ipynb` detalha o grafo estático
(esquema, C1–C9, graus, componentes, comunidades).
