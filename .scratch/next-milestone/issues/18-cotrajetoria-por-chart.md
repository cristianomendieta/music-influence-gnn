# 18 — Cotrajetória separada por chart

**What to build:** a distinção entre Viral 50 e Top 200 nas arestas de cotrajetória passa a chegar ao modelo, hoje ela existe no dado e morre na entrada da rede.

Cada aresta de cotrajetória carrega quatro números em `edge_attr`: dias juntos no chart, distância média de posição, **chart de origem** e primeira semana observada. Só o último é lido, e fora da rede, pela máscara temporal. A convolução recebe apenas `x_dict` e `edge_index_dict`, porque o `SAGEConv` não consome `edge_attr` (`diffusion_gnn.py:134`). Consequência concreta: as duas arestas paralelas de um par que se acompanha nos dois charts viram duas mensagens idênticas, e esse vizinho **entra com peso dobrado na média**. São 610.408 arestas do Top 200, 54.169 do Viral 50, 23.543 pares nos dois.

O conserto barato é separar em **dois tipos de aresta**, `cotrajectory-viral` e `cotrajectory-top`, e deixar o `HeteroConv` dar matrizes próprias a cada um, sem trocar de convolução. De quebra elimina o peso dobrado, porque cada canal passa a ser promediado por conta própria, e dá dois canais a mais para a ablação do item 09 medir. A alternativa ambiciosa, uma convolução que leia `edge_attr` e aproveite também os dias juntos e a distância de posição, pertence ao item 15.

Cabe no mesmo passe a decisão sobre `rev_performs`: hoje `performs` não tem reversa, então o vetor do artista se forma só a partir dos gêneros e do único atributo próprio, nunca a partir do repertório. É a única das quatro políticas de direção que é discutível, e é candidata natural da ablação.

Enquanto isso não existir, a frase correta continua sendo: a distinção existe no dado e está gravada na aresta, ela não chega ao modelo porque a convolução usada não consome atributo de aresta. Contexto em `docs/achados-qualificacao.md`, itens C1, C3 e B6.

**Blocked by:** 06 (a comparação só faz sentido contra a GNN re-treinada na matriz nova)

**Status:** ready-for-agent

- [ ] `cotrajectory` separada em dois tipos de aresta por chart, com matrizes independentes
- [ ] O peso dobrado dos pares presentes nos dois charts deixa de existir
- [ ] `rev_performs` avaliada e a decisão registrada, seja incluí-la ou manter a ausência com justificativa
- [ ] Comparação célula a célula com a GNN do item 06, mesmo protocolo e mesmas seeds
