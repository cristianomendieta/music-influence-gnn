# Pré-compromisso da fase de atenção: o que cada desfecho diz

**Status:** aceito (2026-10-13), antes de qualquer treino com atenção.
Depois do primeiro número, esta página não muda; erros nela viram um ADR novo que a substitui.

## Contexto

A escada de comparação fechou no caso 3 do [ADR-0001](0001-precompromisso-de-falseamento.md): a
GNN com `HeteroGraphSAGE` não supera o baseline sem grafo nem o grafo embaralhado
([resultado](../escada-de-comparacao.md)). Isso admite duas leituras: a estrutura relacional não
agrega neste problema, ou o encoder não sabe aproveitá-la. A segunda é plausível por construção:
o `SAGEConv` **promedia** os vizinhos com o mesmo peso (grau de entrada de até 280 na cotrajetória) e
**ignora `edge_attr`**. O experimento é o que separa as duas leituras. Ele só vale se o desfecho de
cada resultado possível estiver escrito agora, porque rodar variantes depois de um nulo e
reportar a que funcionou é o caminho mais curto para um falso positivo.

## Decisão

Testar **uma** família de encoder com atenção, com hipótese, controles, orçamento de ajuste e
regra de decisão fixados aqui.

### Variante testada

`GATv2Conv` por relação, em torno do mesmo `HeteroConv`, mantendo a cabeça temporal, o protocolo de
splits, a ancoragem na persistência e a prevenção de vazamento.

- Cada nó de tipo `t` recebe `ReLU(W_raiz,t · x + Σ_r GATv2_r(x_origem, x_destino))`. O termo
  `W_raiz` reproduz o ramo raiz do `SAGEConv`, de modo que **sem arestas o modelo pertence à mesma classe funcional
  do sem_grafo** (uma soma de lineares é um linear: o `HeteroConv` do SAGE soma um ramo raiz por relação
  que chega ao tipo, e `W_raiz` é essa soma). É por isso que o sem_grafo existente serve de referência e não se
  retreina. A paridade é de classe funcional, não de dinâmica de treino: a parametrização difere.
- `add_self_loops=False` (as relações com tipos diferentes são bipartidas; o ramo raiz faz o papel
  do laço).
- Atributos de aresta **causais** apenas: `chart` de origem da aresta e `idade = w − first_seen_week`
  (log1p). As colunas `days_together` e `avg_position_distance` de `edge_attr` **ficam de fora**:
  são calculadas sobre o período inteiro (`graph/edges.py::build_cotrajectory` soma todos os dias
  de coexistência), logo incluem dias posteriores à criação da aresta e vazariam o futuro para o
  snapshot da semana `w`. Um teste de invariância é portão de treino.
- HGT **não** é a variante principal: não lê `edge_attr`, não devolve pesos de atenção e devolve
  `None` para tipos sem mensagens (o que quebra o controle sem_grafo). Entra só como robustez
  secundária, se sobrar tempo, e **não decide nada**.

### Controles e matriz

| Modelo | Treinos | Papel |
|---|---|---|
| Atenção sobre o grafo real (`A`) | 2 splits × 3 seeds | a variante |
| Atenção sobre o grafo embaralhado (`A-emb`) | 2 splits × 3 seeds | isola topologia sob o encoder novo ([ADR-0002](0002-grafo-embaralhado-como-controle.md)) |
| sem_grafo (existente) | 0 novos | isola capacidade; é equivalente ao `A` sem arestas |
| Proposta SAGE (existente) | 0 novos | o encoder anterior |

12 treinos novos, mais o ajuste. Mesma config temporal (`W12`), 35 épocas, `max_cotraj_edges = 30_000`
no treino, e avaliação nas **duas** leituras de orçamento (a principal é o grafo completo, por
comparabilidade com o item 10).

### Ajuste de hiperparâmetros (igual para todos)

Grade fixa de 8 configurações: `heads ∈ {2, 4}` × dropout de atenção `∈ {0, 0,2}` × `lr ∈
{5e-4, 1e-3}`, com `hidden = 128` (canais por cabeça = `128/heads`, o que mantém o orçamento perto de
453.121 parâmetros; a diferença exata é declarada). Seleção por `val_mse` no split `current`,
seed 42. A configuração escolhida vale para os dois splits, as três seeds e o `A-emb`. A proposta
SAGE teve 24 configurações; a atenção tem 8, e isso é uma limitação declarada, não uma
vantagem escondida. Nenhuma outra variação é feita depois de ver o teste.

### Hipóteses e família estatística

Em cada split, nas 6 células (2 charts × k ∈ {1, 2, 4}), on-chart:

- **H1:** `A` supera o sem_grafo.
- **H2:** `A` supera o `A-emb`.
- **H3:** `A` supera a proposta SAGE (informativa, não decide).

Família do Holm: as **18 comparações** (3 contrastes × 6 células) de cada split. "Vence" = IC
bootstrap 95% da diferença de erro quadrático por música inteiro abaixo de zero **e** `p` Holm < 0,05.
"Perde" é o espelho. "Maioria" = 4 de 6. Os testes rodam sobre a média das seeds, como no item 10, e
**cada seed isolada é reportada como sensibilidade** (o item 10 escondeu a dispersão).

### O que a dissertação passa a dizer

| Desfecho (por split) | Condição | Afirmação |
|---|---|---|
| **O1** | H1 e H2 em maioria | O nulo era do encoder. Com atenção a topologia paga, e a conclusão da dissertação reverte para o caso 1 (sob este encoder), com o SAGE como limitação |
| **O2** | H1 em maioria, H2 não | O ganho vem de agregar atributos dos vizinhos, não da topologia (caso 2 do ADR-0001) |
| **O3** | H1 não está em maioria, `A` não perde em maioria (inclui H2 sem H1) | O nulo é robusto à arquitetura. A conclusão do ADR-0001 sai fortalecida, e isso é resultado, não fracasso. Se H2 valer sem H1, a nota é que o embaralhado prejudica e o grafo real fica neutro em relação a não ter grafo |
| **O4** | `A` perde para o sem_grafo em maioria | A atenção prejudica (sobreajuste da vizinhança). Reportado como está, com o orçamento de ajuste como limitação |

Precedência: O4 vale antes dos demais. Os quatro desfechos cobrem todas as combinações de H1, H2 e "perde" (a tabela original deixava H2 sem H1 de fora; fechado em 2026-10-18, antes de qualquer treino).

Se os dois splits discordarem, a discordância é o resultado (como no ADR-0005), e a afirmação
principal fica restrita ao que se replica.

### Interpretabilidade (descritiva, pré-declarada)

Pesos de atenção por relação, on-chart, extraídos do `A` e do `A-emb`: entropia normalizada da
atenção por nó de destino em função do grau de entrada. Atenção **não é explicação**
(Jain & Wallace, 2019): fica como descrição e nunca como prova de uso. A medida causal continua
sendo a ablação por tipo de aresta. Previsão pré-declarada e exploratória: o ganho do `A` sobre o
SAGE se concentra nos nós de grau alto de cotrajetória, onde a média dilui (cruzamento com o
estrato P3 da Phase 6b).

## Alternativas rejeitadas

- **HGT como principal.** Rejeitada pelas três razões acima. A atenção por relação do GATv2 com
  `edge_dim` é também a "convolução que lê `edge_attr`" que o ticket 18 deixava para o item 15.
- **Testar várias arquiteturas e reportar a melhor.** Rejeitada: multiplicidade sem correção.
- **Reaproveitar `days_together` e `avg_position_distance`.** Rejeitada: vazamento. Recalculá-las
  "até a semana `w`" é possível, mas custa um pipeline de grafo e fica como extensão pós-resultado.
- **Retreinar o sem_grafo com o encoder novo.** Rejeitada: sem arestas o `A` colapsa no mesmo MLP;
  retreinar só adicionaria ruído de inicialização.

## Consequências

- 12 treinos e a avaliação da matriz custam cerca de 8 a 10 horas de GPU no Colab. A atenção deve
  ser mais lenta que o SAGE (a referência é ~7 min por treino e ~15 min de avaliação por modelo).
- Prazo: matriz concluída até **2026-11-27**. Se não fechar, o que estiver completo é reportado
  como parcial e a lacuna é declarada. Não se corta seed ou controle em silêncio.
- A Phase 6b roda antes e independe desta decisão. Se o mapa mostrar um estrato em que a estrutura
  paga, a leitura de O1/O2/O3 é feita por estrato também, mas isso é análise adicional, fora da regra.
