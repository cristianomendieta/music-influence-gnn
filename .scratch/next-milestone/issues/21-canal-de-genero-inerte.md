# 21 — Sonda do canal de gênero no encoder

**What to build:** a resposta para "por onde o gênero chega ao embedding de música, e chega?". O diagnóstico do item 04 mediu que esvaziar as 9.866 arestas `cooccurs` (gênero↔gênero) ativas na semana 208 **não altera nenhum embedding de música** além do ruído de ponto flutuante (diferença máxima 3e−08; `z_frac_zero`, `z_l2_mean` e `z_std_between_nodes` batem com o grafo completo até a sexta casa).

Com `layers = 3` existe caminho gênero→gênero (camada 1) → gênero→artista (camada 2) → artista→música (camada 3), então o canal deveria propagar algo. Os embeddings saem com ~90% de zeros pós-ReLU, o que sugere que o sinal morre antes de chegar em `music`, mas a causa não foi isolada.

Isso é pré-requisito do item 05: derivar atributos de gênero com fórmula para alimentar um canal que hoje não conduz não move número nenhum. Ver [`docs/diagnostico-ablacao.md`](../../../docs/diagnostico-ablacao.md).

**Blocked by:** 04 (concluído)

**Status:** concluído em 2026-08-31 — **canal utilizável**. Sonda completa (pesos iniciais e treinados) em `docs/sonda-canal-genero.md`. Fica um resíduo herdado para o item 09: sondar `has_genre`/`rev_has_genre`

- [x] Mede a norma e a fração de zeros de `h_genre` e `h_artist` camada a camada, com e sem `cooccurs`
- [x] Diz onde o sinal de gênero morre: **não é ausência de caminho**. Com pesos inicializados o
      canal conduz até `music` (70% dos nós mudam); o que o matava era a agregação — `SAGEConv`
      tira a média dos vizinhos, e a média da tabela aleatória de 530×32, i.i.d. centrada em zero,
      se cancela. Com os atributos do ADR-0003 o sinal que chega em `music` é ~83× maior
- [x] Confirmado com **pesos treinados** (checkpoint re-treinado no grafo do ADR-0003, regime
      `current`): esvaziar `cooccurs` move o embedding de `music` em **1,9e−04**, contra 3e−08 no
      modelo antigo — ~6.200× maior, com 70% dos nós de música alterados. A inércia do item 04 era
      das duas causas somadas: a média da tabela aleatória se cancelava **e** o treino não tinha
      por que preservar o que chegava por ali
- [ ] Verifica o mesmo para `has_genre` e `rev_has_genre`, que hoje têm `delta_rmse` negativo
      (removê-los melhora o modelo) — **passa para o item 09**, junto com a ablação refeita
- [x] Conclui por **canal utilizável**: o item 05 segue como está. Com a ressalva de escala
      registrada — o sinal cai de 24,8% do embedding em `genre` para 0,5% em `music`, e o
      `val_mse` não se mexeu. Conduzir não é ser útil; a utilidade é o que o item 09 mede
