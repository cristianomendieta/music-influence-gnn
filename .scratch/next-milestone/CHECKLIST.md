# Checklist — pós-qualificação

procurar revistas com o tema

## Fase 0 — Diagnóstico (ago) — CONCLUÍDA

- [x] 04 — Diagnóstico da saturação da ablação — **desfecho A**, harness quebrado
      confirmado. Marco desbloqueado. Ver `docs/diagnostico-ablacao.md`

## Fase 1 — Gênero estrutural (ago–set)

- [~] 21 — Sonda do canal de gênero no encoder — o canal conduz; o que o matava era a média de
      uma tabela aleatória (`docs/sonda-canal-genero.md`). Falta repetir com pesos treinados
- [~] 05 — Atributos estruturais de gênero e grafo reconstruído — código entregue e C1-C9 verdes
      nos dois regimes; falta o re-treino no Colab (`notebooks/item05_genero_estrutural_colab.ipynb`)
- [ ] Avaliar pré-processamento de generos.
- [ ] 14 — Ajustes texto

## Fase 2 — Escada de comparação (set–out)

- [ ] 06 — GNN re-treinada na nova matriz
- [ ] 07 — Baseline neural sem grafo
- [ ] 08 — GNN sobre grafo embaralhado (prévia adversa: religar ao acaso melhora o erro;
      igualar o orçamento de arestas entre treino e avaliação)
- [ ] 09 — Ablação por tipo de aresta sobre a GNN final (harness corrigido; refazer
      também a permutação por grupo de features)
- [ ] 10 — Comparação única consolidada

## Fase 3 — Arquitetura com atenção (out–nov)

- [ ] 15 — Arquitetura com atenção (HGT/GAT)
- [ ] 18 — Cotrajetória separada por chart
- [ ] 19 — Cabeça temporal cega ao chart

## Fase 4 — Redação e defesa (dez–jan)

- [ ] Comparação única, discussão, conclusão
