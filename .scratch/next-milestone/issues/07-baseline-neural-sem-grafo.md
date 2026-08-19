# 07 — Baseline neural sem grafo

**What to build:** um preditor recorrente aplicado à série de cada música isoladamente, sem acesso a nenhuma relação do grafo, com as mesmas features de nó e orçamento de parâmetros equivalente ao da proposta, treinado e avaliado na mesma matriz (2 splits × 3 seeds, 2 recortes × 3 horizontes × 2 regimes).

Isola quanto do ganho vem de capacidade neural e quanto vem de estrutura relacional. Responde diretamente à ressalva da banca de que uma RNN bem projetada poderia empatar com a GNN, e é um dos dois braços do pré-compromisso registrado em `docs/adr/0001-precompromisso-de-falseamento.md`.

**Blocked by:** 02, 03, 04

**Status:** ready-for-agent

- [ ] O modelo não acessa nenhuma aresta, e existe teste que garante isso
- [ ] Orçamento de parâmetros comparável ao da proposta, com o número declarado
- [ ] Mesmo protocolo de split, prevenção de vazamento e ancoragem à persistência
- [ ] Seis treinos concluídos e avaliados na mesma matriz da proposta
