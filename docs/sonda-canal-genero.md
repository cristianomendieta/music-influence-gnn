# Sonda do canal de gênero (item 21) — parte arquitetural

Medida em 2026-08-30, CPU local, só inferência. Reprodutível pela seção 2 de
[`notebooks/item05_genero_estrutural_colab.ipynb`](../notebooks/item05_genero_estrutural_colab.ipynb).

**A pergunta.** O item 04 mediu que, com o modelo treinado da qualificação, esvaziar as 9.866
arestas `cooccurs` não alterava nenhum embedding de `music` além de 3e−08. Duas explicações
concorriam: a arquitetura não fecha o caminho gênero→música, ou ela fecha mas o que trafega por
ele é ruído de média zero.

**O que a sonda faz.** Roda o `HeteroSpatialEncoder` (hidden 128, 3 camadas, dropout 0) camada a
camada sobre o snapshot da semana 208, com e sem `cooccurs`, com **pesos apenas inicializados**
(seed fixa: mede a arquitetura, não o que foi aprendido). Duas montagens de `x_genre`: os
atributos estruturais do ADR-0003 e um ruído N(0; 0,1) de 32 dimensões, que é o que a tabela
aprendida era na prática — ela só recebia gradiente pelo caminho gênero→artista→música e ficava
perto da inicialização.

Diferença absoluta média nos embeddings ao esvaziar `cooccurs`:

| camada | tipo | `x_genre` = atributos | `x_genre` = ruído 32d |
|---|---|---|---|
| 1 | genre | 2,9e−01 | 7,6e−03 |
| 2 | artist | 8,6e−02 | 1,4e−03 |
| 3 | **music** | **2,6e−02** | **3,2e−04** |

**Leitura.** O caminho existe e conduz: em três camadas, 70% dos nós de música mudam quando
`cooccurs` é esvaziado, nas duas montagens. O que separa as duas é a **magnitude**: com atributos
estruturais o sinal que chega em `music` é ~83 vezes maior que com a tabela aleatória. A causa é
direta: `SAGEConv` agrega a média dos vizinhos, e a média de vetores i.i.d. centrados em zero é
aproximadamente zero, enquanto atributos correlacionados entre gêneros vizinhos não se cancelam.

**O que isso fecha e o que não fecha.** Fecha a hipótese "não há caminho efetivo": há. Não fecha
o item 21 — a inércia de 3e−08 foi medida com **pesos treinados**, três ordens de grandeza abaixo
dos 3,2e−04 desta sonda com pesos aleatórios, o que indica que o treino também atenuou as
relações de gênero (coerente com o `delta_rmse` negativo dos dois lados do canal). Se o canal
conduz depois de treinado é o que a seção 4 do notebook mede, sobre o modelo re-treinado no grafo
reconstruído.
