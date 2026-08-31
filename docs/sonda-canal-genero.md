# Sonda do canal de gênero (item 21)

Reprodutível pelas seções 2 e 4 de
[`notebooks/item05_genero_estrutural_colab.ipynb`](../notebooks/item05_genero_estrutural_colab.ipynb).
Artefatos: `results/item05_genero_estrutural/sonda_canal_genero_{pesos_iniciais,treinada_current}.parquet`
(fora do git; cópia no Drive).

**A pergunta.** O item 04 mediu que, com o modelo treinado da qualificação, esvaziar as 9.866
arestas `cooccurs` não alterava nenhum embedding de `music` além de 3e−08. Duas explicações
concorriam: a arquitetura não fecha o caminho gênero→música, ou ela fecha mas o que trafega por
ele é ruído de média zero.

**O que a sonda faz.** Roda o `HeteroSpatialEncoder` (hidden 128, 3 camadas, dropout 0) camada a
camada sobre o snapshot da semana 208, com e sem `cooccurs`, e mede a diferença absoluta média
entre os dois embeddings. Com `layers = 3` o caminho existe: `cooccurs` (camada 1) →
`rev_has_genre` (camada 2) → `performs` (camada 3).

## Parte 1 — a arquitetura conduz (pesos apenas inicializados)

Pesos só inicializados, seed fixa: mede a arquitetura, não o que foi aprendido. Duas montagens
de `x_genre`: os atributos estruturais do [ADR-0003](adr/0003-atributos-de-genero-derivados.md) e
um ruído N(0; 0,1) de 32 dimensões, que é o que a tabela aprendida era na prática — ela só recebia
gradiente pelo caminho gênero→artista→música e ficava perto da inicialização.

Diferença absoluta média nos embeddings ao esvaziar `cooccurs`:

| camada | tipo | `x_genre` = atributos | `x_genre` = ruído 32d | razão |
|---|---|---|---|---|
| 1 | genre | 2,4e−01 | 6,7e−03 | 36× |
| 2 | artist | 6,6e−02 | 1,2e−03 | 57× |
| 3 | **music** | **1,7e−02** | **2,6e−04** | **67×** |

**Leitura.** O caminho existe e conduz: em três camadas, 70% dos nós de música mudam quando
`cooccurs` é esvaziado, nas duas montagens. O que separa as duas é a **magnitude**: com atributos
estruturais o sinal que chega em `music` é ~67 vezes maior que com a tabela aleatória. A causa é
direta: `SAGEConv` agrega a média dos vizinhos, e a média de vetores i.i.d. centrados em zero é
aproximadamente zero, enquanto atributos correlacionados entre gêneros vizinhos não se cancelam.

> Uma rodada anterior desta mesma sonda (2026-08-30, CPU local, antes de o build persistir a
> tabela de atributos) deu a mesma ordem de grandeza com razão 83× em vez de 67×. A razão é
> sensível à inicialização; a conclusão qualitativa — uma a duas ordens de grandeza — não é.

## Parte 2 — o modelo treinado usa o canal (pesos treinados)

Mesma sonda, agora com o encoder do checkpoint re-treinado no grafo reconstruído
(`gnn_current_seed42.pt`, regime `current`, `val_mse` 0,000754; ver
[`docs/genero-estrutural-retreino.md`](genero-estrutural-retreino.md)). O snapshot recebe a
popularidade da semana injetada como feature de música, como no treino.

| camada | tipo | dif. média | dif. máx | nós alterados | fração de zeros | dif. relativa¹ |
|---|---|---|---|---|---|---|
| 1 | genre | 1,8e−02 | 6,4e−01 | 87,0% | 62,5% | 13,5% |
| 2 | genre | 2,3e−02 | 3,2e−01 | 87,0% | 49,7% | 20,6% |
| 2 | artist | 1,9e−03 | 2,2e−01 | 81,1% | 81,7% | 5,5% |
| 3 | genre | 2,4e−02 | 2,6e−01 | 97,4% | 49,8% | 24,8% |
| 3 | artist | 8,1e−03 | 1,3e−01 | 81,1% | 45,5% | 15,4% |
| 3 | **music** | **1,9e−04** | 8,6e−02 | **70,3%** | 92,3% | **0,5%** |

¹ diferença média dividida pela magnitude típica de uma coordenada do embedding
(`l2_medio / √128`).

**Leitura.** O canal conduz depois de treinado: 1,9e−04 contra os 3e−08 do modelo antigo, ~6.200
vezes maior e quatro ordens acima do ruído de float, com 70% dos nós de música respondendo. A
inércia medida no item 04 era das duas causas somadas — a tabela aleatória que se cancelava na
média **e** um treino que não tinha por que preservar o que chegava por ali.

A ressalva de escala fica registrada e é o que o item 09 vai ter que interpretar: o sinal de
gênero perde uma ordem e meia de grandeza no salto artista→música (24,8% em `genre`, 15,4% em
`artist`, 0,5% em `music`), coerente com um embedding de música 92% esparso pós-ReLU. Conduzir
não é o mesmo que ser útil, e o `val_mse` praticamente idêntico ao do grafo antigo diz que, por
ora, a contribuição de gênero para o erro é pequena.

## O que isso fecha

**Fecha o item 21 por "canal utilizável"**: das três saídas previstas (canal utilizável, canal a
corrigir, gênero fora do grafo), a medição escolhe a primeira, e o item 05 segue como está.

**Fica aberto:** a mesma sonda para `has_genre` e `rev_has_genre`, que têm `delta_rmse` negativo
no item 04 (removê-los melhora o modelo) — e a ablação por tipo de aresta refeita sobre este
checkpoint, já que o zero de `cooccurs` no item 04 foi medido num canal morto.
