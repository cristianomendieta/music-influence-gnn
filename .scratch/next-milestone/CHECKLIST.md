# Checklist — pós-qualificação

procurar revistas com o tema

## Fase 0 — Diagnóstico (ago) — CONCLUÍDA

- [x] 04 — Diagnóstico da saturação da ablação — **desfecho A**, harness quebrado
      confirmado. Marco desbloqueado. Ver `docs/diagnostico-ablacao.md`

## Fase 1 — Gênero estrutural (ago–set)

- [x] 21 — Sonda do canal de gênero no encoder — **canal utilizável**: conduz com pesos
      iniciais e também depois de treinado (1,9e−04 contra 3e−08 do modelo antigo).
      `docs/sonda-canal-genero.md`
- [x] 05 — Atributos estruturais de gênero e grafo reconstruído — C1-C9 verdes nos dois regimes e
      re-treino feito no regime `current`: `val_mse` 0,000754 contra 0,000749 do grafo antigo.
      `docs/genero-estrutural-retreino.md`. `pre_pandemia` e seeds 43/44 seguem no item 06
- [ ] Avaliar pré-processamento de generos.
- [ ] 14 — Ajustes texto

## Fase 2 — Escada de comparação (set–out)

- [~] 06 — GNN re-treinada na nova matriz — 1 dos 6 treinos feito (`current`/seed 42)
- [ ] 07 — Baseline neural sem grafo
- [ ] 08 — GNN sobre grafo embaralhado (prévia adversa: religar ao acaso melhora o erro;
      igualar o orçamento de arestas entre treino e avaliação)
- [ ] 09 — Ablação por tipo de aresta sobre a GNN final (harness corrigido; refazer
      também a permutação por grupo de features). **Refazer o `cooccurs`**: o zero do item 04
      foi medido num canal de gênero morto. Inclui a sonda de `has_genre`/`rev_has_genre`
      herdada do item 21
- [ ] 10 — Comparação única consolidada

## Fase 3 — Arquitetura com atenção (out–nov)

- [ ] 15 — Arquitetura com atenção (HGT/GAT)
- [ ] 18 — Cotrajetória separada por chart
- [ ] 19 — Cabeça temporal cega ao chart

## Fase 4 — Redação e defesa (dez–jan)

- [ ] Comparação única, discussão, conclusão
