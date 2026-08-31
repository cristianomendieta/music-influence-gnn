# Re-treino sobre o grafo com gênero estrutural (item 05) — resultado

Rodado em 2026-08-31, Colab GPU, via
[`notebooks/item05_genero_estrutural_colab.ipynb`](../notebooks/item05_genero_estrutural_colab.ipynb)
(commit `4a3e7e7`). Fecha a parte experimental do item 05 no regime `current` e responde,
com pesos treinados, a pergunta que o item 21 deixou aberta.

**Condições.** Regime `current`, grafo `hetero_full_current.pt` (gênero = 4 atributos do
[ADR-0003](adr/0003-atributos-de-genero-derivados.md), `train_years` 2017–2019). Config
vencedora da qualificação, `W12_h128_l3_lr5e-04`, dropout 0,2, seed 42, 453.121 parâmetros,
35 épocas em 21,4 min, melhor época a 25ª. Amostras: 321.660 treino / 79.336 val / 192.764
teste. Artefatos em `results/item05_genero_estrutural/` (fora do git; cópia durável no Drive):
`gnn_current_seed42.pt`, `avaliacao_current.parquet`,
`sonda_canal_genero_{pesos_iniciais,treinada_current}.parquet`.

## 1. Custo da nova representação de gênero: nenhum, e nenhum ganho

| grafo | representação de gênero | `val_mse` |
|---|---|---|
| antigo (grid v2, 2026-06-28) | tabela aprendida 530×32 | 0,000749 |
| reconstruído (ADR-0003) | 4 atributos com fórmula | **0,000754** |

Diferença de +0,000005 (+0,7%). A comparação é pareada na configuração (a mesma
`W12_h128_l3_lr5e-04` foi a melhor das 24 da grid v2) e nas features de música e artista —
a remoção de vazamento que as reduziu a 12 e 1 dimensões é de 2026-06-14, anterior à grid v2.
A única variável que muda é a representação de gênero.

A diferença cai **dentro da dispersão** da própria grid v2, cujas 24 configurações ficaram
entre 0,000749 e 0,000764. Com uma seed só não se pode afirmar empate estatístico; o que se
pode afirmar é que trocar 16.960 parâmetros livres por quatro colunas com fórmula **não
degradou o modelo**, e a crítica da banca fica atendida sem custo de desempenho.

## 2. Val e teste contra a persistência: o grafo reconstruído não quebrou nada

Forecasting de um passo, leitura completa (a leitura on-chart e os três horizontes são o item 10).

| split | chart | RMSE GNN | MSE GNN | MSE persistência | GNN vence |
|---|---|---|---|---|---|
| val | viral50 | 0,028971 | 0,000839 | 0,000964 | sim (−13%) |
| val | top200 | 0,024884 | 0,000619 | 0,000861 | sim (−28%) |
| teste | viral50 | 0,025213 | 0,000636 | 0,000716 | sim (−11%) |
| teste | top200 | 0,022083 | 0,000488 | 0,000618 | sim (−21%) |

Os quatro C6/C7 continuam verdes. Não são comparáveis célula a célula com os números do
`summary.md` da Phase 2: aqueles saíram da config fraca `W4_h64_l2_lr1e-03`, e estes da
melhor config — a comparação legítima de níveis é a da seção 1.

## 3. O canal de gênero, depois de treinado

Medição completa em [`docs/sonda-canal-genero.md`](sonda-canal-genero.md), seção "com pesos
treinados". Em uma linha: esvaziar `cooccurs` move o embedding de `music` em **1,9e−04**,
contra os **3e−08** que o item 04 mediu no modelo antigo — quatro ordens de grandeza acima
do ruído de float, com 70% dos nós de música alterados. O canal deixou de ser inerte.

A ressalva de escala fica registrada: 1,9e−04 é **0,5%** da magnitude típica de uma
coordenada do embedding de música, contra 25% em `genre` e 15% em `artist`. O sinal chega,
mas atenuado pelo salto artista→música, e o `val_mse` da seção 1 é consistente com isso —
gênero conduz sem mover a métrica de erro.

## 4. Veredito e o que fica aberto

**Item 21 conclui por "canal utilizável".** As três saídas previstas eram canal utilizável,
canal a corrigir ou gênero fora do grafo; a medição escolhe a primeira, e o item 05 segue
como está. Falta a mesma sonda para `has_genre` e `rev_has_genre`, que têm `delta_rmse`
negativo no item 04.

**Item 05 fica concluído no regime `current`**, com duas pendências que pertencem ao item 06:

- regime `pre_pandemia` (o notebook roda trocando uma variável; `absent_from_network` sobe
  de 20,6% para 30,8% dos gêneros, então a representação muda de verdade entre os regimes);
- seeds 43 e 44, sem as quais o +0,7% da seção 1 não vira afirmação estatística.

**Consequência para a Phase 6.** A ablação por tipo de aresta (item 09) precisa ser refeita
sobre este checkpoint: o `delta_rmse` exatamente 0,000000 de `cooccurs` no item 04 foi medido
num modelo cujo canal de gênero estava morto. Com o canal conduzindo, o número pode mudar —
e se continuar nulo, o achado passa a ser sobre a **utilidade** do gênero, não sobre a
propagação, que é uma afirmação bem mais forte e reportável.
