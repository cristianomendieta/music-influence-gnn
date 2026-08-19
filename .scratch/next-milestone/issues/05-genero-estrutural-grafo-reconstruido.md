# 05 — Atributos estruturais de gênero e grafo reconstruído

**What to build:** gênero deixa de ser uma tabela de 530×32 parâmetros livres aprendidos e passa a ser descrito por atributos com fórmula, derivados da rede gênero↔gênero do MGD+: grau, grau ponderado e número de artistas associados.

A restrição causal é parte da entrega: a rede vem em arquivos anuais e só os anos contidos na janela de treino podem ser usados; as colunas de popularidade e streams médios desses arquivos são agregados de todo o período e vazam o alvo, portanto ficam proibidas. Como o regime de split é configurável, os atributos precisam ser recalculados por regime.

Some-se a isso a remoção de código morto: o grafo hoje carrega features de gênero aleatórias que o modelo sobrescreve.

Ver `docs/adr/0003-atributos-de-genero-derivados.md`.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Atributos de gênero derivados da rede, com fórmula explícita e documentada
- [ ] Só os anos da janela de treino do regime ativo entram no cálculo
- [ ] Colunas que vazam o alvo não são lidas, e existe teste que garante isso
- [ ] Features de gênero aleatórias removidas do grafo
- [ ] Grafo reconstruído e critérios de validação C1 a C9 revalidados
