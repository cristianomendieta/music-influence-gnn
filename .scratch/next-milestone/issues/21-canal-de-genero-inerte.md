# 21 — Sonda do canal de gênero no encoder

**What to build:** a resposta para "por onde o gênero chega ao embedding de música, e chega?". O diagnóstico do item 04 mediu que esvaziar as 9.866 arestas `cooccurs` (gênero↔gênero) ativas na semana 208 **não altera nenhum embedding de música** além do ruído de ponto flutuante (diferença máxima 3e−08; `z_frac_zero`, `z_l2_mean` e `z_std_between_nodes` batem com o grafo completo até a sexta casa).

Com `layers = 3` existe caminho gênero→gênero (camada 1) → gênero→artista (camada 2) → artista→música (camada 3), então o canal deveria propagar algo. Os embeddings saem com ~90% de zeros pós-ReLU, o que sugere que o sinal morre antes de chegar em `music`, mas a causa não foi isolada.

Isso é pré-requisito do item 05: derivar atributos de gênero com fórmula para alimentar um canal que hoje não conduz não move número nenhum. Ver [`docs/diagnostico-ablacao.md`](../../../docs/diagnostico-ablacao.md).

**Blocked by:** 04 (concluído)

**Status:** parte arquitetural respondida em 2026-08-30 (`docs/sonda-canal-genero.md`); falta a medição com pesos treinados

- [x] Mede a norma e a fração de zeros de `h_genre` e `h_artist` camada a camada, com e sem `cooccurs`
- [x] Diz onde o sinal de gênero morre: **não é ausência de caminho**. Com pesos inicializados o
      canal conduz até `music` (70% dos nós mudam); o que o matava era a agregação — `SAGEConv`
      tira a média dos vizinhos, e a média da tabela aleatória de 530×32, i.i.d. centrada em zero,
      se cancela. Com os atributos do ADR-0003 o sinal que chega em `music` é ~83× maior
- [ ] Confirmar com **pesos treinados**: a inércia de 3e−08 do item 04 é 3 ordens abaixo da sonda
      com pesos aleatórios, então o treino também atenuou o canal (seção 4 do notebook do item 05)
- [ ] Verifica o mesmo para `has_genre` e `rev_has_genre`, que hoje têm `delta_rmse` negativo (removê-los melhora o modelo)
- [ ] Conclui por: canal utilizável (segue o item 05 como está), canal a corrigir (a correção entra no item 05) ou gênero fora do grafo (o item 05 muda de escopo)
