# Pré-compromisso: o que a tese passa a dizer se a estrutura não agregar

**Status:** aceito (2026-08-18), registrado antes de rodar qualquer experimento da nova rodada.

A banca de qualificação levantou que uma RNN bem projetada poderia empatar com a
GNN, o que tornaria a contribuição do grafo nula. A tese continua sendo o modelo
proposto, mas passamos a testar sua premissa com o baseline neural sem grafo e o
grafo embaralhado (ADR-0002). Decidimos **agora**, antes de ver o resultado, o que
a dissertação passa a afirmar em cada desfecho:

- **A GNN vence o sem-grafo e o grafo embaralhado:** a tese se sustenta como está,
  e a atribuição do ganho à topologia deixa de ser hipótese e passa a ser medida.
- **A GNN vence o sem-grafo mas não o grafo embaralhado:** o ganho vem de agregar
  atributos de vizinhos, não da topologia. A tese é reescrita nesses termos.
- **A GNN empata com o sem-grafo:** a tese vira condicional. O trabalho passa a
  entregar o mapa de **onde** a estrutura paga (músicas com histórico curto,
  entrada no chart, hits de longa duração multi-onda) e reporta equivalência no
  resto.

O motivo de registrar isto num ADR é que a decisão perde todo o valor se for
tomada depois do número: viraria racionalização. Um resultado nulo, aqui, é
resultado, não fracasso.

## Desfecho (registrado em 2026-10-13)

Acionado em 2026-10-01 sobre a escada completa (18 treinos, 2 splits, 3 seeds). **Caso 3 nos dois
splits.** Detalhes em [`docs/escada-de-comparacao.md`](../escada-de-comparacao.md).

Duas lacunas do texto original, descobertas na aplicação, que os próximos pré-compromissos
precisam fechar de antemão:

1. O ADR diz "empata", mas existe uma célula em que a proposta **perde** (Viral 50, k=4,
   `current`, para o sem_grafo). Passa a valer: o caso 3 cobre "não supera", e uma derrota
   é reportada como derrota.
2. O ADR não fixa a família da correção de Holm. A rodada usou as 24 comparações de cada split
   (inclui SIR e persistência). Com a família restrita às 12 comparações primárias, o desfecho
   não muda (3/6 contra 4/6 exigidos), mas a escolha precisa estar escrita antes.
