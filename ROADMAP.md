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

## O bloqueador, resolvido em 2026-08-30

A ablação por tipo de aresta rodada em jul/2026 devolveu variação de erro
**exatamente zero** para os cinco tipos de aresta e para os três grupos de features.
O diagnóstico ([docs/diagnostico-ablacao.md](docs/diagnostico-ablacao.md)) fechou por
**desfecho A**: o zero exato era instrumento quebrado, e o clamp satura por cima disso.

O harness da ablação encodava a semana alvo, que a predição nunca lê: 0% das posições
da janela chegavam ao GRU, e `Δ` era a constante 0,0032 para as 98.186 amostras. Com o
harness corrigido, a ablação move o erro. E o recorte on-chart é o que devolve
sensibilidade: na leitura completa o clamp anula 76,9% da correção estrutural, no
on-chart anula 3,6%.

Os números liberam o marco, mas trazem três coisas que o plano precisa absorver:

- **todo o sinal está na cotrajetória** — sem ela o RMSE on-chart sobe 93%; os demais
  tipos de aresta têm efeito **adverso** (removê-los reduz o erro);
- **o canal de gênero está inerte** — esvaziar as 9.866 arestas gênero↔gênero não muda
  nenhum embedding além do ruído de float. Investigar antes de investir na fase de
  gênero estrutural;
- **religar a cotrajetória ao acaso melhora o erro** — prévia adversa do controle de
  topologia, confundida pelo descasamento treino (30k arestas subamostradas) contra
  avaliação (grafo completo).

`results/phase3/interpretability.parquet` mede uma constante e não pode ser citado.

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

Somam-se as divergências texto ↔ código recuperadas das sessões de preparação da
qualificação, consolidadas em [docs/achados-qualificacao.md](docs/achados-qualificacao.md):
a fórmula da cabeça temporal no documento, três correções pontuais em slides e specs,
e as limitações de arquitetura que hoje só existem como resposta oral.

## Trabalho fatiado

21 tickets em [.scratch/next-milestone/issues/](.scratch/next-milestone/issues/),
com as dependências entre eles declaradas. Os itens 16 a 20 entraram em 2026-08-19,
vindos dos achados da preparação da qualificação.

Podem começar já: **01** (recorte on-chart), **02** (regime de split parametrizado),
**03** (estatística correta), **11** (EDA da imputação), **12** (glossário no texto),
**13** (dataset na introdução), **16** (fórmula da cabeça temporal no documento),
**17** (correções pontuais em slides e specs), **20** (limitações de arquitetura no texto).

Caminho crítico: **01 → 04 → 21 → 05 → 06 → {08, 09} → 10**. O item **21** (sonda do
canal de gênero) entrou em 2026-08-30, vindo do diagnóstico. Os itens **18** e **19**
(cotrajetória por chart, cabeça sensível ao chart) pendem de **06** e são variações de
arquitetura, não pré-requisito do argumento central.
