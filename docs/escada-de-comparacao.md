# Escada de comparação — resultado (itens 06 a 10)

**Fechado em 2026-10-01** (execução dos notebooks), **registrado em 2026-10-13**.
Desfecho do [ADR-0001](adr/0001-precompromisso-de-falseamento.md): **caso 3, a tese vira
condicional**, nos dois regimes de split.

Fontes: `notebooks/item06_08_escada_treinos_colab.ipynb` (treinos),
`notebooks/item09_avaliacao_matriz_colab.ipynb` (Modo 2 e ablação),
`notebooks/item10_comparacao_consolidada.ipynb` (comparação e veredito),
`notebooks/resultados_escada_rapido.ipynb` (leitura rápida).
Artefatos da comparação em `results/comparacao_final/` (parquets; o Modo 2 por modelo e a
ablação, de `avaliacao_matriz/`, estão só no Drive).

## Protocolo executado

18 treinos: {proposta, sem_grafo, embaralhado} × {`current`, `pre_pandemia`} × seeds {42, 43, 44},
config `W12_h128_l3_lr5e-04`, **453.121 parâmetros nas três variantes**. As variantes saem de
`graph/controls.py`, então só a topologia muda. Avaliação no Modo 2, k ∈ {1, 2, 4}, nos dois
charts, leitura on-chart como principal ([ADR-0004](adr/0004-recorte-on-chart-como-leitura-principal.md)).
Estatística: erro quadrático médio por música, Wilcoxon pareado, IC bootstrap da diferença,
Holm sobre as 24 comparações de cada split. "Vence" = IC inteiro abaixo de zero **e** `p` Holm < 0,05.

## Resultado

### Treino (`val_mse`, leitura completa, 1 passo)

| Split | proposta | embaralhado | sem_grafo |
|---|---|---|---|
| `current` | **0,000752** ± 0,000002 | 0,000766 ± 0,000003 | 0,000783 ± 0,000002 |
| `pre_pandemia` | **0,000906** ± 0,000004 | 0,000913 ± 0,000004 | 0,000911 ± 0,000002 |

A proposta tem o menor erro de validação nas 6 combinações de split e seed. No `current` a
margem sobre o sem_grafo é de 4%; no `pre_pandemia` é de 0,5% (0,000906 contra 0,000911), dentro
da ordem do desvio entre seeds.

### Modo 2, on-chart, RMSE médio das 6 células (chart × k)

| Split | Ordenação (melhor → pior) |
|---|---|
| `current` | sem_grafo 0,16673 < persistência 0,16869 < embaralhado 0,17158 < proposta 0,17341 < SIR 0,25133 |
| `pre_pandemia` | proposta 0,14989 < sem_grafo 0,15155 < persistência 0,15235 < embaralhado 0,15487 < SIR 0,23333 |

As ordenações **não se preservam** entre os splits: o confundidor que o
[ADR-0005](adr/0005-segundo-split-pre-pandemia.md) queria responder responde que o resultado
depende do período.

### Veredito do ADR-0001

| Split | Vence o sem_grafo | Vence o embaralhado | Perde para o sem_grafo | Desfecho |
|---|---|---|---|---|
| `current` | 0/6 | 0/6 | 1/6 (viral50, k=4) | caso 3 |
| `pre_pandemia` | 1/6 (viral50, k=4) | 1/6 (viral50, k=4) | 0/6 | caso 3 |

Contra o SIR a proposta vence 5/6 nos dois splits. Contra a persistência, 1/6 (e perde 1/6)
nos dois.

O que cada número diz:

- **O SIR está fora de disputa.** Todo modelo neural o bate com folga.
- **A persistência é dura.** Nenhum modelo neural a supera de forma consistente no Top 200 em k=1 e k=2.
- **A topologia, sozinha, não explica ganho.** Proposta e embaralhado ficam a menos de 2%
  um do outro no `current` e, no `pre_pandemia`, a proposta melhora 1% a 5,4% sobre o embaralhado
  e só passa da regra em uma célula.
- **O sem_grafo é competitivo ou melhor no `current`.** No Viral 50, k=4, ele vence a proposta
  por 9% de RMSE, com IC da diferença de erro quadrático [+0,0136; +0,0200] e `p` Holm < 0,001.

### Ablação por tipo de aresta (Δ RMSE on-chart ao remover, média de 3 seeds)

| Componente | `current` | `pre_pandemia` |
|---|---|---|
| sem `cotrajectory` | **+0,0688** | **+0,0447** |
| permutar acústicas | +0,0029 | +0,0030 |
| sem `cooccurs` | +0,00007 | +0,00020 |
| permutar metadados | +0,00002 | +0,00009 |
| sem `has_genre` | −0,00040 | −0,00045 |
| sem `rev_has_genre` | −0,00145 | −0,00145 |
| sem `performs` | −0,00276 | −0,00135 |

Nenhum delta é exatamente zero: o instrumento mede (fecha o item 04). `cooccurs` mede **~zero
com o canal vivo**, então o achado sobre gênero passa a ser de utilidade, não de propagação.

## Leituras que o texto precisa respeitar

1. **A ablação mede dependência, não necessidade.** Remover a cotrajetória em tempo de avaliação
   tira o modelo da distribuição de treino (+0,069). O sem_grafo, treinado sem nenhuma aresta,
   empata. A informação da cotrajetória não é necessária para o desempenho, só é *usada* pelo
   modelo que a viu no treino.
2. **O `val_mse` favorece a proposta e o Modo 2 on-chart não.** São medidas diferentes (1 passo
   com teacher forcing, leitura completa, contra rollout recursivo no recorte on-chart). A
   discrepância é um achado a explicar, não a esconder.
3. **O caso 3 do ADR diz "empata", mas no `current` o sem_grafo vence uma célula.** O ADR não
   cobre o "perde". O notebook trata qualquer não-vitória como caso 3. A conclusão não muda,
   mas a redação deve dizer "não supera, e em uma célula perde".
4. **A família do Holm.** As 24 comparações de cada split incluem SIR e persistência. Como
   sensibilidade, pós-hoc e **sem alterar o veredito**: com Holm só sobre as 12 comparações
   contra sem_grafo e embaralhado, a proposta vence o embaralhado em 3/6 no `pre_pandemia`
   (1/6 com a família de 24), o sem_grafo em 1/6, e no `current` perde para o embaralhado
   em 2/6. A regra de maioria exige 4/6, então o desfecho é o mesmo.
5. **Wilcoxon e bootstrap respondem coisas diferentes.** O primeiro é sobre postos, o
   segundo sobre a média das diferenças. Em 11 das 48 células on-chart o IC exclui zero e o
   `p` não, ou o contrário, e em 7 o sinal da média e o da taxa de vitória discordam. Isso é
   diferença pesada na cauda, não defeito: a regra exige os dois, o que é conservador.
6. **Os testes rodam sobre a média das 3 seeds, não sobre cada seed.** O notebook do item 10
   promedia `y_pred` entre as seeds antes de calcular o erro. Isso tem dois efeitos: (a) os
   modelos neurais entram como um comitê de 3, contra um único SIR e uma única persistência, o
   que os favorece nessa comparação; (b) a dispersão entre seeds sai do teste. Ela existe: o desvio
   do RMSE on-chart entre seeds tem mediana 0,0027 e máximo 0,027, do mesmo tamanho que boa
   parte das diferenças entre as variantes neurais. A comparação entre as três variantes
   neurais continua justa (todas recebem o mesmo tratamento), mas "vence a persistência" é
   mais fraco do que a tabela sugere.

## Pendências herdadas

- A leitura de orçamento casado de arestas (`teto_treino`, ticket 08) foi rodada só na seed 42;
  nas cinco linhas visíveis no notebook (embaralhado, `current`, leitura completa) o
  `teto_treino` dá erro ligeiramente **maior** (+0,00003 a +0,0005), o que não sustenta que o
  grafo completo desfavoreça o modelo real. As 72 linhas completas estão só no Drive.
- ADR-0001 exige, no caso 3, o **mapa de onde a estrutura paga**. É a feature
  [`phase-6b-mapa-estrutura`](../.specs/features/phase-6b-mapa-estrutura/spec.md).
- A hipótese de que o encoder SAGE (média simples) limite o resultado é a que a
  [Phase 7](../.specs/features/phase-7-atencao/spec.md) testa, com pré-compromisso no
  [ADR-0006](adr/0006-precompromisso-fase-atencao.md).
