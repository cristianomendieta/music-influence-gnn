# Grafo embaralhado como controle de topologia

**Status:** aceito (2026-08-18)

O baseline neural sem grafo, sozinho, **não** isola a contribuição da topologia. Se
a GNN vencer esse baseline, resta a explicação de que ela venceu por agregar
atributos de artistas e gêneros vizinhos, e não porque a estrutura das relações
importa. As duas arquiteturas também diferem em capacidade, o que confunde ainda mais
a leitura.

Por isso a escada de comparação inclui um terceiro elemento: a **mesma** GNN,
mesmas features, mesma capacidade e mesmo protocolo, treinada sobre um grafo com
as arestas religadas aleatoriamente preservando distribuição de grau e tipos de
aresta. Só a topologia muda. A diferença entre a GNN real e a embaralhada é, por
construção, atribuível à estrutura.

A ablação por tipo de aresta complementa isso respondendo **qual** relação carrega
o sinal, e roda em tempo de avaliação, sem re-treino.

**Consequência:** a escada final é persistência, SIR, sem-grafo, GNN embaralhada e
GNN completa, com ablação por aresta sobre a completa.
