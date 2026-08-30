# Diagnóstico da ablação zerada — resultado (item 04)

Rodado em 2026-08-30, Colab GPU (T4), via
[`notebooks/item04_diagnostico_saturacao_colab.ipynb`](../notebooks/item04_diagnostico_saturacao_colab.ipynb).
Fecha o bloqueador registrado em `.specs/project/STATE.md` e na Phase 4 do
[ROADMAP detalhado](../.specs/project/ROADMAP.md).

**Condições.** Checkpoint `W12_h128_l3_lr5e-04` (W = 12, `val_mse` = 0,000749), regime de
split `current`, 27 semanas alvo do span de teste (stride 2), 98.186 amostras sobre 1.955
músicas, 4,6% on-chart e 95,4% no piso. Sem treino: só inferência. Artefatos brutos
(`componentes.parquet`, `variantes.parquet`, `ablacao.parquet`, `embeddings.parquet`,
`DIAGNOSTICO.md`) em `MyDrive/music-influence-gnn/item04_diagnostico/`, fora do git.

## Veredito: desfecho A, com a hipótese C confirmada

O zero exato de julho/2026 **não era saturação**: era instrumento quebrado. A ablação
daquela rodada não mediu o grafo, mediu uma constante.

| Montagem do banco de embeddings | desvio de Δ | Δ mín | Δ máx | valores distintos |
|---|---|---|---|---|
| janela `[w−12, …, w−1]` (correta) | 0,023389 | −0,443126 | 0,022936 | 36.314 |
| semana alvo `w` (harness antigo) | **0,000000** | 0,003204 | 0,003204 | **1** |

O harness antigo encontrava **0%** das posições da janela no banco; todas caíam no ramo
`torch.zeros(B, hidden)`, o GRU recebia sequência nula e Δ virava a constante 0,00320397
para todas as 98.186 amostras. Com a montagem correta, 100% das posições são encontradas.
Corrigido em `evaluation/interpretability.py` (commit `2c645c6`).

Com o instrumento consertado, a ablação move o erro, e a hipótese A também se sustenta:
o clamp é o que mata a sensibilidade **na leitura completa**.

## O clamp: quanto da correção sobrevive

| leitura | n | `y_prev` médio | \|Δ\| médio | Δ / predição | clamp ativo | correção anulada |
|---|---|---|---|---|---|---|
| todas | 98.186 | 0,014892 | 0,010728 | 0,720 | **82,8%** | **76,9%** |
| on-chart (principal) | 4.543 | 0,282608 | 0,023222 | 0,082 | **5,3%** | **3,6%** |
| piso | 93.643 | 0,001904 | 0,010122 | 5,317 | 86,5% | 85,1% |

Na leitura completa o clamp engole três quartos da correção estrutural. No recorte on-chart
engole 3,6%. [ADR-0004](adr/0004-recorte-on-chart-como-leitura-principal.md) previu
exatamente isso: o recorte on-chart não é escolha de apresentação, é o que devolve
sensibilidade à medição.

Por chart, na leitura completa: `top200` RMSE 0,023369 contra 0,022856 da persistência (o
modelo **perde**); `viral50` 0,025083 contra 0,026385 (ganha). Coerente com o que a
qualificação já reportava.

## O grafo altera a predição antes do clamp: sim

| variante | Δ muda | dif. média de Δ | dif. máx | predição muda (pós-clamp) | on-chart | piso |
|---|---|---|---|---|---|---|
| sem `performs` (artist→music) | 99,1% | 0,005872 | 0,191074 | 33,9% | 94,4% | 31,0% |
| sem `has_genre` (artist→genre) | 97,3% | 0,001556 | 0,108060 | 16,7% | 91,1% | 13,1% |
| sem `rev_has_genre` (genre→artist) | 97,3% | 0,002782 | 0,132864 | 18,2% | 91,7% | 14,6% |
| sem `cotrajectory` (music→music) | 86,9% | 0,019303 | 0,371949 | 82,4% | 93,3% | 81,9% |
| sem `cooccurs` (genre→genre) | 30,6% | **0,000000** | 3e−08 | 0,7% | 8,1% | 0,4% |
| `cotrajectory` religada ao acaso | 100,0% | 0,004734 | 0,424919 | 17,4% | 95,3% | 13,7% |

## A ablação refeita — `delta_rmse` por leitura

| componente | Δ RMSE completa | Δ RMSE on-chart | Δ RMSE pré-clamp | RMSE on-chart |
|---|---|---|---|---|
| grafo completo (referência) | 0 | 0 | 0 | 0,100487 |
| sem `performs` | −0,000095 | −0,002506 | −0,005544 | 0,097981 |
| sem `has_genre` | −0,000054 | −0,000763 | −0,002497 | 0,099724 |
| sem `rev_has_genre` | −0,000104 | −0,001623 | −0,003598 | 0,098864 |
| sem `cotrajectory` | **+0,018048** | **+0,093292** | +0,025133 | 0,193779 |
| sem `cooccurs` | 0,000000 | 0,000000 | 0,000000 | 0,100487 |
| `cotrajectory` religada ao acaso | −0,000150 | −0,002840 | −0,008468 | 0,097647 |

A escala do recorte importa: para `performs`, o efeito on-chart é 26 vezes maior que na
leitura completa. Era essa a assinatura prevista para o desfecho A.

## Quatro achados que mudam o planejamento

**1. Todo o sinal estrutural está na cotrajetória.** Removê-la leva o RMSE on-chart de
0,1005 para 0,1938, um aumento de 93%. Nenhum outro tipo de aresta chega perto.

**2. Os outros quatro tipos de aresta pioram o modelo.** Todos os `delta_rmse` fora da
cotrajetória são **negativos**: remover artista→música, os dois lados de gênero e religar
a cotrajetória ao acaso **reduz** o erro on-chart. Não é ausência de efeito, é efeito
adverso.

**3. `cooccurs` (gênero↔gênero) é inerte.** Esvaziar as 9.866 arestas ativas na semana 208
não altera nenhum embedding além do ruído de float (3e−08): `z_frac_zero`, `z_l2_mean` e
`z_std_between_nodes` batem com o grafo completo até a sexta casa. Com três camadas existe
caminho gênero→gênero→artista→música no encoder, então o canal deveria propagar algo. A
causa não foi isolada (o encoder devolve embeddings com ~90% de zeros pós-ReLU, o que
sugere que o sinal morre antes de chegar em `music`). Isso é pré-requisito do item 05:
não faz sentido derivar atributos de gênero para alimentar um canal que hoje não conduz.

**4. A cotrajetória religada ao acaso não piora nada — melhora.** É a prévia barata do
item 08, e ela é adversa à tese: o que o modelo usa parece ser o **canal** de agregação de
vizinhos, não a topologia particular. Com uma ressalva forte que impede conclusão: o
treino roda com `max_cotraj_edges = 30_000` ([trainer.py:48](../src/music_diffusion_gnn/training/trainer.py#L48))
e a avaliação usa o grafo completo, 480k–664k arestas. O modelo nunca viu a topologia real
inteira durante o treino, viu um subconjunto aleatório por snapshot. Um religamento
aleatório é, para ele, mais próximo da distribuição de treino do que o grafo completo. O
veredito sobre topologia continua sendo o item 08, com treino sobre grafo embaralhado.

## O que isso invalida

`results/phase3/interpretability.parquet` (2026-07-07) mede uma constante. Os zeros exatos
por tipo de aresta e por grupo de features de lá não são resultado, são artefato do
harness, e não podem ser citados na dissertação. A permutação por grupo de features tem o
mesmo defeito e precisa ser refeita junto com o item 09.
