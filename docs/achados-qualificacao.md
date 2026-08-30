# Achados da preparação da qualificação (10–12/08/2026)

Consolidado a partir das sessões de estudo que antecederam a apresentação
(transcrições `a1362405`, `798757d1`, `fad0670a`). Registra o que foi apurado
lendo o código e medindo o grafo/checkpoint, sobretudo onde **texto, slides e
código não dizem a mesma coisa**. Complementa o tracker do marco seguinte em
`.scratch/next-milestone/issues/`, que nasceu depois, na sessão de 18/08.

Legenda de status: **aberto** · **decidido manter** · **resolvido** · **coberto por issue NN**

---

## A. Divergências documento/slides ↔ código

| # | Onde | O que o texto diz | O que o código faz | Status |
|---|---|---|---|---|
| A1 | `documento_qualificao/4-metodologia/texto.tex:125` e slide "Cabeça temporal" | `ŷ = 0,5·σ(MLP(h))` | cabeça residual: `Δ = MLP(h)`, `ŷ = clamp(y_prev + Δ, 0, 0,5)` (`diffusion_gnn.py:184-193`) | slide **resolvido** (12/08); **documento aberto**, §4.7 ainda tem a fórmula antiga |
| A2 | `documento_qualificao/4-metodologia/texto.tex:38`, specs Fase 1, slide 23 | ausentes imputados pela **mediana** | imputa **0 no espaço padronizado**, que é a **média** (`nodes.py:100-104`) | **decidido manter** (11/08) para slide e documento não divergirem entre si; relacionado à issue 11 |
| A3 | Slide "Etapas 3 a 5", `alertblock` | limites da normalização são "teóricos e fixos, **não** o máximo empírico da série" | `smoothed.min()/max()` por série, é empírico (`preprocess.py:29-31`) | **decidido manter** (12/08, revertido a pedido). Cuidado de fala: o bloco é usado como argumento de ausência de vazamento; não apontar para ele e afirmar isso em voz alta |
| A4 | Descrição do Modo 2 (Wilcoxon) | "pareado por **música**" | pareava por **origem de previsão** (~23.291 pares em vez de 1.933), o próprio comentário do código chamava de *"per-song" proxy* | **resolvido** no código em 18/08 (issue 03). Pendência viva: a **faixa de p do texto** (6,5×10⁻⁴ a 5,4×10⁻³²) não bateu com nenhuma recomputação |
| A5 | `.specs/features/phase-1-hetero-graph/design.md:17` | reversa vem do `ToUndirected()` do PyG | feita à mão com `flip(0)` (`build.py:141`) | **aberto**, sem gravidade, `docs.md` já descreve certo |
| A6 | Slide 26 | "tabela 530×32 otimizada junto com os pesos: **a rede descobre a geometria**" | a medição contradiz a segunda metade (ver B4) | **aberto**, decisão de redação pendente |
| A7 | Slide 20, diagrama de pré-processamento | seis etapas | são **sete**: falta a **densificação** (calendário diário contínuo, todo dia fora do chart vira **zero**, `preprocess.py:89-93`) | **aberto**. Não é detalhe: é a origem direta do fato B1 |
| A8 | Fala antiga do slide 15 | GraphSAGE "agrega por amostragem de vizinhança" | o forward é sobre o **grafo inteiro** da semana; a única poda é o DropEdge no `cotrajectory` (teto 30.000), que é regularização | **resolvido** na fala |
| A9 | Justificativa de `L = 3` | apresentada como raciocínio (2 é o mínimo para gênero chegar na música, 3 dá folga) | escolhido por **busca em grade**; o raciocínio explica por que 3 venceu, não motivou testar | **resolvido** na fala. Dizer na ordem certa evita parecer racionalização a posteriori |
| A10 | Bloco do quadro 27 | "cotrajetória não tem reversa" de forma absoluta | a precedência é definida **por chart**; **5.292 pares** (22% dos que aparecem nos dois charts, 0,8% do total) têm direções **opostas** nos dois | **resolvido** na fala, com a vírgula |

## B. Achados de medição que não estão em documento nenhum

**B1. Composição do conjunto de teste.** ~**95%** das semanas-alvo do teste são de
música **fora do chart**, no valor de piso (95,3–95,4% para k = 1, 2, 4). Só 510
músicas têm alguma semana em chart em 2021. As estreias se espalham: 62% das
músicas (1.214) aparecem em **um único ano civil**, só 3 aparecem nos cinco.
Permanência mediana de 4 semanas por par música-chart. Não invalida a comparação
(mesmas semanas, teste pareado), invalida a leitura de "o erro é pequeno".
→ endereçado pela issue 01 (recorte on-chart no Modo 2).

**B2. On-chart quantificado no Modo 1.** No espectro completo o SIR erra 89%
(Viral 50) e 84% (Top 200) menos; no recorte on-chart, **9%** e **30%** menos. Os
erros sobem de ~0,03 para ~0,19. Boa parte da vantagem do SIR vem de **descrever
bem o silêncio**, não a subida e a descida.

**B3. Clustering contra o acaso.** `cotrajectory`: 641.034 pares, 6.526 nós, 2.378
isolados, C médio 0,5126, densidade 0,0301, razão **17,0×**. O subgrafo de gêneros
dá razão 17× também (0,81 sem os isolados). Calculado do `hetero_full.pt`, não
está em nenhum arquivo do repo.

**B4. O `genre_emb` colapsou no checkpoint avaliado.** Em
`results/phase2_experimentos_v2/grid_best_model.pt`: **514 das 530 linhas com norma
< 0,001**, 63 exatamente zero, maior valor absoluto 0,0029 (na inicialização cada
linha tinha norma ≈ 0,566). As matrizes que transportam gênero foram junto (norma
0,000 na camada 1). Teste direto: trocar a tabela treinada por ruído muda os
embeddings de música em **7×10⁻⁶ relativos**. Na prática o canal de gênero **não
contribui** para a previsão. Causas não separadas: `weight_decay` de 1e-5 sobre uma
tabela a três saltos da perda, ou gênero é mesmo pouco informativo.
→ motiva as issues 05 e 09. Armadilha de checkpoint: `results/phase2/best_model.pt`
é o modelo **fraco** (W4/h64/L2, sem `genre_emb`); o avaliado é o `grid_best_model.pt`.

**B5. Média móvel × agregação semanal.** Autocorrelação lag-1 semanal em 1.498
pares: **0,867 com** média móvel, **0,878 sem**. No eixo semanal ela praticamente
não muda nada. A composição de dois filtros de 7 dias é uma janela triangular de 13.

**B6. Cotrajetória por chart.** 610.408 arestas vêm do Top 200, 54.169 do Viral 50;
23.543 pares aparecem nos dois. A direção da seta é **da música que entrou primeiro
no chart para a que entrou depois** (`edges.py:217-226`).

**B7. Distribuição de grau.** Cotrajetória: máximo 280 (entrada) / 2.311 (total),
p90 238/568, mediana 30/56, 2.381 músicas com grau zero. Cuidado: a mediana do
`stats.md` é sobre os nós que **têm** aquela aresta (197 de entrada); contando as
6.526 músicas cai para 30. Os dois números estão certos, muda o denominador.

**B8. Comunidades de Louvain** (só no subgrafo de coocorrência de gêneros):
136 = bloco brasileiro, 119 = rap/rock anglófono, 96 = pop alternativo/indie,
60 = eletrônica, 43 = latino, **7 = família soul inteira** (classic/neo/northern/
southern soul, motown, quiet storm). Proibido: citar valor de modularidade (não é
calculado), dizer que a partição é estável (seed 42, estabilidade não testada), ou
chamar as comunidades de "os gêneros verdadeiros".

**B9. `genre_dim = 32` nunca foi buscado.** É o default de `diffusion_gnn.py:72`; a
grade variou só janela, dimensão oculta, camadas e taxa de aprendizado.
530 × 32 = 16.960 parâmetros.

**B10. O teto do DropEdge vale na validação.** O MSE de validação de 7,49×10⁻⁴, que
**selecionou a configuração**, foi medido com grafo podado. Só a avaliação final
roda completa (`eval_cotraj = None`, `run_phase2.py:202`).

## C. Limitações de arquitetura reconhecidas

**C1. `edge_attr` nunca chega ao modelo.** As quatro colunas da aresta de
cotrajetória (dias juntos, distância média de posição, **chart**, primeira semana)
existem no arquivo, mas o `SAGEConv` não consome `edge_attr` — só o `x_dict` e o
`edge_index_dict` entram (`diffusion_gnn.py:134`). A primeira semana é lida fora da
rede, pela máscara. Consequência: um vizinho que acompanha a música **nos dois
charts entra com peso dobrado na média**. Conserto barato: separar em dois tipos de
aresta (`cotrajectory-viral`, `cotrajectory-top`) e deixar o `HeteroConv` dar
matrizes próprias a cada um. Frase certa: *"a distinção existe no dado e está
gravada na aresta; ela não chega ao modelo porque a convolução que eu uso não
consome atributo de aresta"*.

**C2. O Δ é cego ao chart.** A sequência que entra na GRU não tem dimensão de chart;
o código do chart é usado **uma vez só**, para buscar a âncora `y_prev`
(`diffusion_gnn.py:190-193`). Para a mesma música e semana, **o Δ do Viral 50 e o do
Top 200 é o mesmo número**, e as duas previsões diferem apenas pela âncora. Visível
em cinco linhas de código. Variação oferecível: separar a cabeça por chart, ou
passar o chart como entrada dela.

**C3. `performs` sem reversa.** O artista nunca recebe mensagem do repertório: o
vetor dele se forma só a partir dos gêneros e do único atributo próprio. Artista com
dez faixas em alta e artista com dez faixas mortas ficam igualmente descritos.
`rev_performs` é candidata natural da ablação. Formulação: escolha de projeto que
tomaria diferente hoje, não esquecimento.

**C4. Ponderação por relação é global, não por caso.** `aggr="sum"` no `HeteroConv`
(`encoder.py:36`): nada olha a vizinhança daquela música e redistribui importância.
Adaptatividade por caso só existe no eixo do **tempo**, nos portões da GRU. Frase:
*o modelo é adaptativo no tempo e fixo nas relações*. É a lacuna que o HGT preencheria
(issue 15).

**C5. Média joga fora o grau.** Uma música com 2 predecessoras e outra com 200
entregam vetores de escala parecida, e nenhum atributo de nó guarda a contagem.
Contrapartida: é a média que torna coerente treinar podado (30k arestas) e avaliar
completo (664.577).

**C6. Indutivo no tempo, transdutivo nos nós.** Música e artista entram por
atributos, mas **gênero é transdutivo** (embedding por identidade). E as 6.526
músicas do teste já estão todas no grafo — a divisão é por semana. Não dizer "eu
avalio em músicas que o modelo nunca viu"; dizer "a arquitetura permitiria isso".
Cold start não é medido, e é severo: antes de estrear a música é **nó isolado**
(o `first_seen_week` de `performs` é a primeira semana dela em chart), sem
cotrajetória e com âncora zero, então o modelo degenera para um MLP sobre os
atributos acústicos.

**C7. Ablação por tipo de aresta implementada e nunca executada** à época
(`interpretability.py:66`; a Fase 3 rodou com `--skip-interpret`, não existe
`results/phase3/interpretability.parquet`). Hoje é a issue 09, e o resultado zero
exato virou o bloqueador registrado em `ablation_zero`. **Fechado em 2026-08-30:** o
zero era o harness encodando a semana alvo em vez da janela, e o `Δ` chegava constante
ao GRU. Ver [`diagnostico-ablacao.md`](diagnostico-ablacao.md). Os números de
`results/phase3/interpretability.parquet` não podem ser citados.

## D. Afirmações a não fazer (correções de fala)

1. **"A GNN vence todos os baselines"** — não. Contra o SIR, 6/6. Contra a
   persistência, Viral 50 em todos os *k*, Top 200 só em *k* = 4.
2. **"No Modo 1 o teste é 2021"** — errado. O Modo 1 **não tem divisão**: modelo
   congelado percorrendo a vida inteira de cada música, o que **inclui semanas de
   treino**. Por isso nenhuma afirmação do trabalho se apoia no Modo 1. A divisão
   (182 / 207 / 208) aparece uma vez só, no treino da Fase 2; o Modo 2 a respeita.
3. **"A semente é a semana 12"** — são as **12 primeiras semanas de vida de cada
   música**, por par música-chart (`rollout.py:87-90`). Vida menor encurta a semente;
   menos de duas semanas exclui. Daí a diferença entre 1.981 músicas no ajuste SIR e
   1.955 no Modo 1.
4. **"O rollout é livre"** — livre **nos valores, não na estrutura**: o grafo
   continua sendo o real, mascarado até a semana anterior ao alvo, e ele cresce
   durante a reconstrução. Não é vazamento, mas é vantagem que o SIR não tem.
5. **"Os atributos de aresta entram no modelo"** — não entram (C1).
6. **"A rede descobriu a geometria dos gêneros"** — a medição diz o contrário (B4).
   Descrever a **decisão de projeto**, não o resultado.
7. **"Lei de potência" / "rede livre de escala"** — exige ajustar a distribuição e
   testar. Dizer "a distribuição de grau é muito desigual".
8. **"Com três camadas eu olho três vizinhos"** — camada conta **saltos**, não
   vizinhos.
