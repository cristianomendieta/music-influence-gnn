# Handoff — Apresentação da qualificação de mestrado

**Data:** 2026-08-09 · **Branch:** `main` · **Escopo:** slides Beamer + guia de estudo

---

## Contexto

Cristiano Mendieta defende o exame de qualificação de mestrado (PPGInf/UFPR, agosto de 2026),
orientado por André Luís Vignatti. Tema: modelagem da difusão de tendências culturais com redes
neurais heterogêneas temporais em grafos, aplicada a charts musicais brasileiros do Spotify.

Duas tarefas foram pedidas nesta sessão:

1. **Slides da apresentação** (45 min), em LaTeX, seguindo o layout de `slides-beamer-ufpr/`.
2. **Material de estudo** (artifact) cobrindo toda a apresentação, com aprofundamento nos tópicos
   que ele marcou à mão no PDF entregue à banca.

Ambas foram entregues. O que segue é o estado, as decisões e o que ficou pendente.

---

## Fontes usadas

| Arquivo | Papel |
|---|---|
| `documento_qualificao/` | fonte LaTeX do documento (note o typo no nome da pasta: `qualificao`, não `qualificacao`) |
| `documento_qualificao/{1-intro,2-fundam,3-arte,4-metodologia,5-resultados,6-cronograma}/texto.tex` | todo o conteúdo dos capítulos |
| `Modelagem_da_difusão_de_tendências_culturais_com_r_260808_185651.pdf` | versão entregue à banca, **com as anotações manuscritas** |
| `slides-beamer-ufpr/` | template institucional (main.tex + 3 PNGs) |

### Como reler as anotações do PDF

**Não são annots de PDF.** `pypdf` e `pymupdf` retornam zero annotations — as marcações estão
achatadas como imagens dentro do conteúdo das páginas. Para relê-las é preciso renderizar e olhar:

```python
import pymupdf
d = pymupdf.open('Modelagem_da_difusão_de_tendências_culturais_com_r_260808_185651.pdf')
for i in [2,5,6,10,16,17,18,19,20,21,24,28]:      # páginas com marcação
    d[i-1].get_pixmap(dpi=110).save(f'/tmp/p{i:02d}.png')
```

`pymupdf` não estava instalado no venv; foi adicionado com
`uv pip install pymupdf --python .venv/bin/python` (o venv não tem `pip`).

---

## As 20 marcações do PDF (todas já cobertas)

| Pág. | Trecho marcado | Anotação |
|---|---|---|
| 2 | "leitura interpretável" (resumo) | *por quê?* |
| 2 | "HeteroGraphSAGE por snapshots semanais com cabeça GRU" | grifado |
| 2 | "estrutura topológica não aleatória, clustering elevado…" | *saber explicar isso* |
| 5 | "tais modelos tendem a operar sobre populações agregadas" | *o que são?* |
| 6 | Q3 — interpretabilidade epidemiológica preservada | *importante saber justificar isso* |
| 6 | parágrafo do valor aplicado (curadoria, investimento, antecipação) | *bom parágrafo para os slides* |
| 6 | "homofilia" | *saber explicar esse termo* |
| 10 | snapshots discretos vs. TGAT/TGN | *saber explicar a decisão e se faz sentido a outra abordagem* |
| 16 | §4.3 inteira | *slide bem feito sobre as etapas do pré-processamento* |
| 16 | normalização (rank-score, MA-7d, min-max, piso) | *saber bem sobre essa normalização* |
| 16 | "indicador binário de ausência acústica" | *como é feito esse indicador binário?* |
| 16 | escore-z com estatísticas só do treino | *importante saber explicar bem* |
| 17 | embeddings de gênero, 32 dim, aprendidos | grifado |
| 18 | escolha de snapshots por custo e grão temporal | *saber justificar* |
| 18 | desacoplamento codificador / cabeça temporal | *saber explicar de forma didática* |
| 19 | §4.7.1 codificador espacial heterogêneo | *explicação didática e entender fórmulas* |
| 19 | §4.7.2 snapshots e banco de embeddings | *explicação didática* |
| 20 | dropout / weight decay / early stopping | grifado |
| 21 | Tabela 4.1 de hiperparâmetros | *estudar hiperparâmetros* |
| 24 | "detecção de comunidades por Louvain" | *saber explicar* |
| 28 | baseline neural sem grafo | *entender e justificar* |

---

## Entrega 1 — Slides

**Arquivo:** `apresentacao/main.tex` · PNGs da UFPR copiados para a mesma pasta.

- Beamer 16:9, tema `default` + `orchid`, layout e página de título herdados de `slides-beamer-ufpr`.
- **49 quadros no fonte** (48 de conteúdo + 1 template de transição) → com as 7 seções, ~55 slides
  renderizados.
- **Todos os diagramas em TikZ/pgfplots.** Nenhuma imagem externa além dos 3 PNGs institucionais —
  decisão deliberada, porque `results/` é gitignored e as figuras da Fase 3 não existem em disco.
- Rodapé com altura fixa (`beamercolorbox` com `ht`/`dp`) para o conteúdo não colidir com o logo
  UFPR e os losangos do fundo institucional.

### Estrutura

```
Capa
Roteiro (gerado por \AtBeginSection)
1. Introdução                     → Introdução · Em uma frase
2. Motivação e problema           → curva de chart · valor aplicado · SIR · população agregada ·
                                    hipótese · pergunta de pesquisa · contribuições
3. Fundamentação teórica          → IC/LT · homofilia · grafo heterogêneo · DTDG vs CTDG ·
                                    passagem de mensagens · famílias de arquiteturas
4. Trabalhos relacionados e lacuna→ tabela das três linhas · diagrama da lacuna
5. Dados e metodologia            → desenho experimental · dados · pré-processamento (4 slides) ·
                                    SIR · grafo (nós/arestas) · snapshots · vazamento ·
                                    arquitetura · codificador (2) · banco · DropEdge · GRU ·
                                    treino · protocolo · dois regimes
6. Resultados parciais            → reprodução SIR · topologia · Modo 1 · Modo 2 · leitura · discussão
7. Próximos passos e cronograma   → quatro frentes · cronograma · mensagens principais
```

### Diagramas construídos

curva de rank-score (pgfplots) · compartimentos S→I→R · duas curvas idênticas (limitação do SIR) ·
grafo música–artista–gênero · IC vs LT · homofilia (dois clusters) · esquema de tipos do grafo ·
DTDG vs CTDG · passagem de mensagens em 2 camadas · diagrama de Venn da lacuna · pipeline
experimental · pipeline de pré-processamento em 6 etapas · vetor de 12 dimensões com a flag `miss` ·
snapshots monotônicos · arquitetura de 2 módulos com gradiente de retorno · banco de embeddings ·
timeline dos splits · barras do Modo 2 (pgfplots) · Modo 1 vs Modo 2.

---

## Entrega 2 — Guia de estudo (artifact)

**URL:** https://claude.ai/code/artifact/2bdfa98c-40e3-4d1a-a7f8-50d060f15ba5 · favicon 🎓

> ⚠️ O HTML vive no scratchpad da sessão, que é **efêmero** — não sobrevive à sessão seguinte.
> **Para atualizar, passe a URL acima como parâmetro `url` da ferramenta Artifact**, senão um novo
> link é criado. Para recuperar o conteúdo numa sessão nova: `WebFetch` na URL — ela salva o HTML
> completo em `tool-results/`; basta remover o wrapper `<!doctype html><html><head>…<body>` do
> frame-runtime (linha 1) e o `</body></html>` final antes de editar.

Conteúdo: mesma ordem da apresentação, com bloco âmbar em cada uma das 20 marcações (citação do PDF +
página + explicação longa + *frase pronta* para dizer em voz alta), blocos rosa nas armadilhas,
perguntas de banca em `<details>`, fichas de números para decorar e glossário.

### Revisão de 2026-08-09 — camada didática (2ª sessão)

Feedback em duas rodadas. **(1)** *"faltam detalhes nas explicações… o que significa SIR, EDO etc.;
deve ser conteúdo para eu ler e saber de tudo, de forma didática."* **(2)** Depois de ler a primeira
tentativa (uma `Parte 0` upfront): *"prefiro que as explicações sejam encaixadas nos temas a que
pertencem; dessa forma os conceitos ficam um pouco perdidos."*

**Decisão final: conceito no ponto de uso, nunca upfront.** A Parte 0 foi dissolvida e cada bloco
plantado imediatamente antes do trecho que o consome. Resultado: 140 KB → 185 KB.

Novo bloco visual **`.conceito`** (fundo `--surface-2`, filete esquerdo `--ink-3`, tag mono
"CONCEITO") — 9 ocorrências. Distinto do âmbar (`.marcado` = marcação do PDF) e do rosa
(`.trap` = armadilha). Onde cada um foi parar:

| Conceito | Onde vive agora |
|---|---|
| modelo / parâmetro / hiperparâmetro / ajuste / graus de liberdade | Parte 1, `#modelo`, logo antes do SIR |
| SIR letra por letra + Kermack-McKendrick 1927 | Parte 1, `#sir`, entre o diagrama S→I→R e as EDOs |
| derivada, EDO palavra por palavra, integrar = simular (Euler, RK45) | Parte 1, `#sir`, logo depois das EDOs |
| β*SI* como ação das massas · conservação · pico em *S* = 1/R₀ · onda única | Parte 1, `#sir` |
| mínimos quadrados não lineares (o laço de 4 passos) | Parte 1, `#sir`, fecha a subseção |
| *G* = (*V*, *E*), grau, vizinhança 𝒩ᵣ(v), salto, dirigido, atribuído, densidade | Parte 2, `#vocab-grafo` — **primeira** subseção da Fundamentação |
| neurônio, MLP, por que a não linearidade é obrigatória, ReLU/sigmoide/tanh, notação **h**ᵥ⁽ˡ⁾ | Parte 2, `#rede-neural`, logo antes da GNN |
| RNN → gradiente que desaparece → GRU (portões r e z, fórmulas) + vs. LSTM/Transformer | Parte 3, `#gru`, dentro de "A cabeça temporal" |
| perda/MSE, MSE·RMSE·MAE, gradiente, descida, retropropagação, autograd, época/minibatch, sobreajuste, 3 splits | Parte 3, `#treino` (h3 renomeado para "Como a rede aprende…") |
| fórmula do clustering local — explica de onde vem o 0,5126 | Parte 4, `#topologia`, antes dos números |
| teste de hipótese, *p*-valor, Wilcoxon vs. Mann-Whitney vs. teste t | Parte 4, `#stats`, antes da tabela do Modo 2 |

Fora dos `.conceito`:

- **`#siglas`** — 25 siglas por extenso (PT + EN + uma linha), agora como `<h2>` na área **Apoio**,
  imediatamente antes do Glossário. É material de consulta, não de leitura linear.
- **3 SVGs novos**, nos mesmos tokens: retas tangentes (derivada) · curvas treino/validação com
  early stopping · célula GRU com os portões.
- **8 perguntas de banca novas** num bloco *Sobre fundamentos* (total 32) — as "básicas" que
  derrubam: o que é SIR, o que é EDO/integrar, como β e γ são estimados, o que é GRU e por que não
  LSTM/Transformer, o que uma camada de GNN faz, por que não linearidade, o que é gradiente e
  retropropagação, por que 3 splits.
- **Glossário de 18 → 55 verbetes**, alfabético. Agora é consulta rápida; a explicação longa de cada
  termo vive no `.conceito` correspondente.
- **Leitura da fórmula do codificador em voz alta** sobre uma música concreta (artista via
  `performs` + 40 vizinhas via `cotrajectory`), dentro do bloco marcado de §4.7.1.
- **Ordem de estudo em 4 passadas** no "Como usar" — a 1ª lê os Conceitos, a 2ª pula.
- Adam explicado por extenso dentro de "Perda e otimizador".

Design: um único token/classe novo (`.conceito`); todo o resto reusa a paleta e as classes
existentes. Anchors e balanceamento de tags validados por script (`quebrados: []`, nenhum
desbalanceamento). Backup da versão com Parte 0 em `backup-parte0.html`, no scratchpad efêmero.

---

## Entrega 3 — Guia de dados e metodologia (artifact)

**URL:** https://claude.ai/code/artifact/62457cc0-4706-4f14-b178-33cf4230ee86 · favicon 🔬 ·
~172 KB · fonte no scratchpad efêmero (`guia-metodologia.html`)

Pedido em 2026-08-09: *"artifact exclusivamente focado em dados e metodologia… as motivações, os
porquês de cada decisão… deve conter também uma seção de possíveis perguntas da banca."*

**Diferença essencial em relação aos outros dois materiais:** este foi escrito **lendo a
implementação**, não só o capítulo 4. Fontes: `src/music_diffusion_gnn/**`, `scripts/run_phase*.py`,
`tests/`, `EXPERIMENTS.md`, além de `documento_qualificao/4-metodologia/texto.tex`.

Formato: **registro de decisões**. 26 blocos `D-01`…`D-26`, cada um com quatro campos fixos —
*O que · Alternativas · Por quê · Custo*. Mais 15 blocos `.codigo` (violeta) com o trecho real do
repositório, 4 blocos `.diverge` (ocre) e 45 perguntas de banca em `<details>`.

Seções: desenho experimental (4 fases + artefatos + condição de passagem) · fontes de dados e os
dois universos · pré-processamento (6 etapas) · grafo heterogêneo (nós, 5 relações, regras de
construção reais, C1–C7) · temporalidade e vazamento (3 portas, semana ISO, splits) · modelo
(codificador fórmula a fórmula, R1, cabeça) · treino · protocolo de avaliação (Modo 1 e a armadilha
OQ1, Modo 2, métricas e estatística) · consolidado das divergências · banca · números.

### ⚠️ As 4 divergências documento ↔ código encontradas

Registradas também na memória `divergencias-documento-codigo`. Resumo:

| # | Ponto | Sev. | Ação |
|---|---|---|---|
| 1 | **R1 (2026-06-23)** não está no §4.7: `pop_bank` injeta 2 canais de popularidade defasada nas features de música (12→14 dims dinâmicas) e a cabeça emite resíduo com `ŷ = clamp(y_prev + Δ, 0, 0.5)`, zero-init → modelo não treinado **reproduz a persistência**. Documento descreve `ŷ = 0,5·σ(MLP)`. Não é vazamento (semana injetada ≤ alvo−1). | alta | reescrever §4.7, ou mencionar a revisão você mesmo |
| 2 | **min-max é empírico por série** (`smoothed.min()/max()` em `data/preprocess.py`), enquanto §4.3 afirma limites teóricos e fixos. Toda música atinge 0,5 no próprio pico; o valor da semana 3 depende do pico da semana 120. | alta | corrigir código e reexecutar, **ou** corrigir o parágrafo e declarar limitação |
| 3 | **`SAGEConv` ignora `edge_attr`** — só topologia é usada. Papel do artista, dias em conjunto, peso etc. estão construídos mas não entram no modelo. Exceção: `first_seen_week`, usado pelo `mask_until`. | média | uma frase em §4.5 |
| 4 | `has_genre` tem `first_seen_week = 0` para todas as arestas; `cooccurs` usa proxy anual e **sobrescreve atributos com o ano mais recente**. Inócuo por causa do ponto 3. | baixa | nota de rodapé |

### Achados de implementação que não estavam em nenhum documento

- **cotrajectory:** limiar de **≥ 7 dias** de coocorrência; até 2 arestas paralelas por par (uma por
  chart); direção pela ordem de entrada no chart, desempate lexicográfico; `first_seen_week` = semana
  do **7º dia acumulado**, não do primeiro encontro.
- **Universo do grafo:** `top200_songs ∪ (viral50_songs ∩ songs_with_features)` — assimétrico de
  propósito, porque incluir todo o Viral 50 daria **>48% de imputação** nesse grupo.
- **Início da série por música:** `max(release_date, 2017-01-01)`. É o que explica a diferença entre
  os RMSE de 0,0381/0,0699 (primeira execução, registrada em `EXPERIMENTS.md`) e 0,0289/0,0471.
- **Semana ISO não é bijetora** (2020 tem 53 semanas ISO): 2020-06-30 e 2020-07-01 colidem em 182;
  2020-12-31 e 2021-01-01 em 208. Resolvido fixando os splits por índice, com val estritamente entre.
- **Modo 1 usa rollout livre, não teacher forcing** — o docstring de `evaluation/rollout.py` nomeia
  isso como OQ1: com teacher forcing puro, a cabeça residual lê `y_prev` real e degenera em
  persistência, vencendo o SIR trivialmente.
- **Dropout não é aplicado após a última camada** do codificador (o embedding que vai ao banco é limpo).
- **`_zscore` dos atributos de nó roda sobre todo o universo**, não só o treino — mas os atributos
  são estáticos, então não há distribuição temporal a vazar. Os que exigiriam split foram removidos.
- **`test_node_feature_leakage.py`** trava a ausência de agregados full-series com um teste
  comportamental: injeta atividade futura no chart e exige que o vetor do nó não mude.

---

## Feedback recebido e o que foi feito

| Feedback | Ação |
|---|---|
| "gostei muito dos slides" | — |
| "em alguns slides tem texto demais… menos texto, o suficiente para eu olhar e explicar" | Reescrita completa: bullets viraram frases de 3–8 palavras, parágrafos dentro de `block` viraram uma linha, prosa explicativa migrou para o artifact. Diagramas, tabelas e fórmulas mantidos. |
| "remova esses slides backup" | `\appendix`, os 4 quadros de backup e a `thebibliography` removidos. |
| "falta um slide de introdução" | Slide de Introdução adicionado após a capa. |
| "não gostei do desenho do slide de introdução, talvez a introdução deveria estar dentro do roteiro" | Timeline de fases (TikZ) descartada; virou `description` de 4 rótulos sem desenho. `Introdução` promovida a `\section`, e o quadro "Roteiro" avulso foi removido (o `\AtBeginSection` já cobre). |

### Princípio de design a manter

**Slide = gatilho visual, não texto corrido.** A explicação profunda vive no artifact. Se um slide
novo precisar de mais de ~4 bullets curtos ou de um parágrafo, o conteúdo provavelmente pertence ao
guia de estudo, não ao slide.

---

## ⚠️ Limitação importante

**Os slides nunca foram compilados.** Não há LaTeX nesta máquina:

- `pdflatex`, `xelatex`, `lualatex`, `latexmk`, `kpsewhich` — todos ausentes
- `tectonic` — ausente
- `docker` existe (binário do Docker Desktop via WSL) mas **o daemon não está rodando**

A validação feita foi estática, por script Python: balanceamento de `\begin`/`\end` por ambiente e
contagem de chaves ignorando escapes e comentários. **Resultado: zero desbalanceamentos.**

### Primeira coisa a fazer na próxima sessão

```bash
cd apresentacao && pdflatex main.tex && pdflatex main.tex   # 2x, por causa do sumário
```

Pontos a conferir visualmente no PDF gerado, por ordem de risco:

1. **Larguras de TikZ.** Vários diagramas usam coordenadas absolutas em cm. Área útil do slide 16:9 é
   ~12,8 cm. Os maiores: pipeline de pré-processamento (até x≈10,4 + largura do bloco), snapshots
   monotônicos (até x≈10,5), timeline dos splits (12,0), arquitetura (x≈10,0).
2. **`\usebackgroundtemplate` com `FundoUFPR2.png`.** O fundo tem logo no canto inferior esquerdo e
   losangos no inferior direito. Verificar se nenhum slide encosta neles.
3. **`pgfplots` com `compat=newest`.** Se a distribuição TeX for antiga, pode reclamar.
4. **Overfull hboxes** nas tabelas de `Trabalhos relacionados` e `Modo 2`.
5. **Contagem final de slides** e cronometragem contra os 45 min.

---

## Estado do repositório

- `apresentacao/` — **novo, não commitado**: `main.tex`, `CapaUFPR.png`, `FundoUFPR.png`,
  `FundoUFPR2.png`, `HANDOFF.md`
- `slides-beamer-ufpr/` — template original, intocado
- `documento_qualificao/` — **não modificado** nesta sessão
- `scripts/run_phase3.py` — modificado (`M`), de trabalho anterior, **não relacionado** a esta sessão
- `apresentacao.zip` e o PDF na raiz — não versionados

Nada foi commitado. Nenhum arquivo do usuário foi apagado ou sobrescrito fora de `apresentacao/`.

---

## Pendências / próximos passos possíveis

- [ ] **Compilar e revisar visualmente** (bloqueador — ver acima)
- [ ] Ensaiar cronometrando; ~55 slides em 45 min dá ~50 s por slide
- [ ] Se sobrar tempo no ensaio, candidatos a corte: *Famílias de arquiteturas*,
      *Difusão em redes: os dois modelos clássicos*
- [ ] Decidir se `apresentacao/` entra no git (hoje não está no `.gitignore`)
- [ ] Opcional: gerar as figuras reais da Fase 0/1/3 (boxplot da Fig. 3, distribuições de grau,
      comunidades Louvain) e substituir alguns diagramas esquemáticos por dados reais.
      Ver a memória `results-gitignored-phase0-not-on-disk` — exigiria rodar `run_phase0.py` de novo.

---

## Notas úteis para quem continuar

- O documento marca várias figuras como **"(adiada)"** nos comentários LaTeX — elas nunca foram
  produzidas. Por isso os slides usam diagramas esquemáticos, não plots de dados reais.
- `STATUS.md` está **desatualizado** (diz que a Fase 1 é a próxima; na verdade a Fase 3 já rodou).
  Não use como fonte de estado.
- Números que aparecem em slides e guia, para conferência cruzada: 6.526 músicas · 1.701 artistas ·
  530 gêneros · 1.981 no subconjunto *viral*∩*hit* · arestas 9.274 / 3.344 / 664.577 / 9.866 ·
  clustering 0,5126 e 0,6040 · MSE de validação 7,5×10⁻⁴ · W=12, d=128, L=3.
- Memórias relacionadas do projeto: `qualificacao-apresentacao`, `phase3-evaluation-spec`,
  `results-gitignored-phase0-not-on-disk`, `phase2-training-notebook`.
