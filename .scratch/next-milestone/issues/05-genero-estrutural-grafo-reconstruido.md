# 05 — Atributos estruturais de gênero e grafo reconstruído

**What to build:** gênero deixa de ser uma tabela de 530×32 parâmetros livres aprendidos e passa a ser descrito por atributos com fórmula, derivados da rede gênero↔gênero do MGD+: grau, grau ponderado e número de artistas associados.

A restrição causal é parte da entrega: a rede vem em arquivos anuais e só os anos contidos na janela de treino podem ser usados; as colunas de popularidade e streams médios desses arquivos são agregados de todo o período e vazam o alvo, portanto ficam proibidas. Como o regime de split é configurável, os atributos precisam ser recalculados por regime.

Some-se a isso a remoção de código morto: o grafo hoje carrega features de gênero aleatórias que o modelo sobrescreve.

Ver `docs/adr/0003-atributos-de-genero-derivados.md`.

**Ressalva de 2026-08-30.** O diagnóstico do item 04 mostrou que esvaziar as 9.866 arestas
gênero↔gênero não altera nenhum embedding de música além do ruído de float, e que os dois
lados do canal de gênero têm `delta_rmse` negativo (removê-los melhora o modelo). Enquanto
o item 21 não disser por onde o gênero chega em `music`, atributos novos não movem número
nenhum: a entrega desta issue passa a incluir a evidência de que o canal conduz.

**Blocked by:** 04 (concluído), 21

**Status:** concluído em 2026-08-31 no regime `current`. Resultado em `docs/genero-estrutural-retreino.md`. O regime `pre_pandemia` e as seeds 43/44 passam para o item 06, que já os pede na matriz

- [x] Atributos de gênero derivados da rede, com fórmula explícita e documentada
      (`graph/nodes.py::genre_attributes`, ADR-0003)
- [x] Só os anos da janela de treino do regime ativo entram no cálculo
      (`SplitRegime.train_years`; um grafo por regime: `hetero_full_{regime}.pt`)
- [x] Colunas que vazam o alvo não são lidas, e existe teste que garante isso
      (`tests/test_genre_features.py`; `Avg_*` fora também do `edge_attr` de `cooccurs`)
- [x] Features de gênero aleatórias removidas do grafo, junto com a tabela 530×32 do modelo
      (checkpoints anteriores a ADR-0003 não carregam mais, por desenho)
- [x] Grafo reconstruído e critérios de validação C1 a C9 revalidados nos dois regimes
- [x] Evidência de que o canal conduz — parte arquitetural: `docs/sonda-canal-genero.md`
- [x] Evidência de que o canal conduz **depois de treinado**: 1,9e−04 no embedding de `music`
      contra 3e−08 do modelo antigo (`docs/sonda-canal-genero.md`, parte 2). Item 21 fechado
- [x] Re-treino da config vencedora sobre o grafo reconstruído: `val_mse` **0,000754** contra
      0,000749 do grafo antigo (+0,7%, dentro da dispersão 0,000749–0,000764 da grid v2). Trocar
      16.960 parâmetros livres por quatro colunas com fórmula não custou desempenho. Bate a
      persistência nos quatro pares split×chart (`docs/genero-estrutural-retreino.md`)
