# Roadmap — da qualificação à defesa

Resumo de uma página. Plano detalhado em [.specs/project/ROADMAP.md](.specs/project/ROADMAP.md),
decisões com justificativa em [docs/adr/](docs/adr/), vocabulário em [CONTEXT.md](CONTEXT.md).

**Qualificação aprovada em ago/2026. Defesa prevista para jan/2027.**

## A pergunta do marco

A dissertação afirma que o MusicDiffusionGNN supera o SIR e a persistência. A banca
levantou que uma rede recorrente bem projetada poderia empatar com ele, o que
tornaria a contribuição do grafo nula.

A tese continua sendo o modelo proposto, mas passa a ser testável: este marco mede
quanto do ganho vem da **estrutura relacional** e quanto vem apenas da flexibilidade
de um modelo neural.

O que a dissertação passa a dizer em cada desfecho já está registrado, antes de ver
qualquer número, em [ADR-0001](docs/adr/0001-precompromisso-de-falseamento.md).

## As seis decisões

| | Decisão |
|---|---|
| 1 | A tese é mantida, mas com teste da premissa. Se o baseline sem grafo empatar, ela vira condicional: o trabalho entrega o mapa de **onde** a estrutura paga |
| 2 | Entram na comparação um **baseline neural sem grafo** e a mesma GNN sobre um **grafo embaralhado**. O segundo é o que isola topologia; o primeiro sozinho não isola |
| 3 | Gênero deixa de ser tabela de 530×32 parâmetros livres e passa a ter atributos com fórmula, derivados da rede gênero↔gênero restrita aos anos de treino |
| 4 | O **recorte on-chart** vira a leitura principal, porque ~95% dos alvos de teste são semanas fora do chart, no valor de piso |
| 5 | Entra um **segundo split, inteiramente pré-pandemia**, como checagem de robustez. O split atual testa 100% em período de pandemia |
| 6 | 3 seeds por modelo, IC bootstrap sobre a **diferença**, Wilcoxon reagregado por música, correção de Holm |

## O bloqueador

A ablação por tipo de aresta já rodou, em jul/2026, e devolveu variação de erro
**exatamente zero** para os cinco tipos de aresta e para os três grupos de features.
Predições bit a bit idênticas com e sem grafo.

A hipótese é que o instrumento sature: a predição é `clamp(y_prev + Δ, 0, 0,5)`, o
grafo entra só por `Δ`, e nas semanas de piso qualquer `Δ` negativo é anulado pelo
clamp. O desfecho alternativo é que o grafo realmente não influencie a predição, e
aí o ganho sobre o SIR vem do ancoramento à persistência.

**Nenhum treino novo antes desse diagnóstico.**

## As fases

| Fase | O que entrega | Quando |
|---|---|---|
| Diagnóstico | Veredito sobre a ablação zerada | ago |
| Gênero estrutural | Novos atributos e grafo reconstruído | ago–set |
| Escada de comparação | Persistência, SIR, sem-grafo, grafo embaralhado, proposta | set–out |
| Arquitetura com atenção | HGT ou GAT no lugar do HeteroGraphSAGE | out–nov |
| Redação e defesa | Comparação única, discussão, conclusão | dez–jan |

**Matriz experimental:** 3 modelos neurais × 2 splits × 3 seeds = 18 treinos, mais
SIR e persistência refeitos, avaliados em 2 recortes × 3 horizontes × 2 regimes.

## Em paralelo, sem depender de número novo

Pendências de texto levantadas pela banca: glossário aplicado ao resumo e à
metodologia, dataset e suas limitações na introdução, caracterização justa do SIR,
fluxograma do processamento dos dados, EDA que sustente a imputação por mediana, e
a seção 4.7 separando proposta de fundamento.

## Trabalho fatiado

15 tickets em [.scratch/next-milestone/issues/](.scratch/next-milestone/issues/),
com as dependências entre eles declaradas.

Podem começar já: **01** (recorte on-chart), **02** (regime de split parametrizado),
**03** (estatística correta), **11** (EDA da imputação), **12** (glossário no texto),
**13** (dataset na introdução).

Caminho crítico: **01 → 04 → 05 → 06 → {08, 09} → 10**.
