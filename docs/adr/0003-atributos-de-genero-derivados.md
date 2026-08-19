# Gênero deixa de ser tabela de parâmetros livres

**Status:** aceito (2026-08-18), substitui a representação usada na qualificação.

Até aqui, gêneros eram representados por uma tabela aprendida de 530×32 parâmetros
livres, inicializada aleatoriamente e otimizada fim a fim. A banca cobrou três
coisas sobre isso, e as três procedem: não há fórmula que se possa escrever na
metodologia, a representação não é relacional (é uma identidade aprendida), e a
tabela nunca foi explicada no texto. Havia ainda código morto: o grafo carregava
features de gênero aleatórias que o modelo sobrescrevia.

Gênero passa a ser descrito por atributos com fórmula, derivados da rede
gênero↔gênero do MGD+: grau, grau ponderado e número de artistas associados.

**Restrição causal:** a rede vem em arquivos anuais, e só os anos contidos na janela
de treino podem ser usados. As colunas `Avg_Popularity` e `Avg_Streams` desses
mesmos arquivos são agregados de todo o período e **vazam o alvo**; ficam proibidas.

**Alternativa rejeitada:** remover o tipo de nó gênero. Eliminaria a crítica por
deleção, mas descartaria o canal de difusão mais plausível do grafo e esvaziaria a
ablação por tipo de aresta.

**Consequência:** o grafo precisa ser reconstruído e todos os modelos re-treinados;
os resultados anteriores a esta decisão não são comparáveis aos novos.
