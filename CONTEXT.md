# Difusão de popularidade musical em grafo temporal heterogêneo

Glossário da linguagem do projeto. A banca de qualificação apontou termos mal
definidos no resumo e na metodologia; este arquivo é a fonte única de vocabulário.
Ele não descreve implementação nem estado do projeto.

## Dados e universo

**MGD+**:
Dataset público de charts, atributos acústicos e redes de gênero do mercado
musical, do qual este trabalho usa o recorte brasileiro. É um dataset, não um
ecossistema.
_Evitar_: ecossistema de dados do MGD+, base MGD+.

**Chart**:
Uma das duas listas diárias do Spotify Brasil usadas aqui: Viral 50 e Top 200.
Cada música gera uma série independente por chart.
_Evitar_: ranking, lista, regime (regime é outra coisa, ver abaixo).

**Regime**:
Qual dos dois charts está sendo medido. Viral 50 mede viralidade; Top 200 mede
sucesso. Os dois regimes compartilham modelo e protocolo, e diferem só nos dados.
_Evitar_: baseline (não existem dois baselines, existe um baseline medido em dois regimes).

**Subconjunto viral∩hit**:
As 1.981 músicas que aparecem nos dois charts no período, universo de avaliação
herdado do trabalho replicado.

**Escore de posição**:
A série alvo, obtida da posição no chart invertida, suavizada por média móvel de
7 dias e normalizada para [0; 0,5].
_Evitar_: popularidade, streams (a série não mede nenhum dos dois).

**Piso**:
O valor atribuído às semanas em que a música não está no chart. Não é uma
observação de popularidade baixa, é ausência de observação.
_Evitar_: valor mínimo, zero.

**Recorte on-chart**:
Restrição da avaliação às semanas em que a música está efetivamente no chart,
excluindo as semanas no piso. É a leitura principal dos resultados, porque a
leitura sem recorte é dominada por acertar ausência.

## Grafo

**Grafo de influência**:
O grafo heterogêneo música–artista–gênero sobre o qual a propagação é aprendida.
_Evitar_: rede social, grafo de colaboração.

**Co-trajetória**:
Aresta música→música que existe quando duas músicas coexistiram no chart por pelo
menos sete dias. É uma relação observada, e a data de criação da aresta é a data
dessa sétima coexistência.
_Evitar_: similaridade, correlação.

**Grafo embaralhado**:
O mesmo grafo com as arestas religadas aleatoriamente, preservando distribuição de
grau e tipos de aresta. Serve como controle: mantém features, capacidade e
agregação constantes, e destrói só a topologia.
_Evitar_: grafo aleatório, grafo nulo.

**Ablação por tipo de aresta**:
Remoção de um tipo de relação do grafo, medindo a variação de erro que resulta.
Mede de qual relação vem o sinal.

## Modelos

**Baseline populacional**:
O modelo SIR clássico ajustado por música e por regime, herdado do trabalho
replicado. É um único baseline, avaliado em dois regimes e em dois modos.
_Evitar_: baselines populacionais, ambos os baselines.

**Persistência**:
Baseline ingênuo que prevê que o escore da próxima semana é igual ao da semana
atual. É o piso de comparação mais duro do problema.

**MusicDiffusionGNN**:
O modelo proposto: codificador espacial heterogêneo sobre o grafo de influência,
seguido de cabeça temporal recorrente, prevendo uma correção sobre a persistência.
_Evitar_: GNN temporal, o modelo (quando houver ambiguidade com as variantes).

**Baseline neural sem grafo**:
Preditor recorrente aplicado à série de cada música isoladamente, com as mesmas
features de nó e orçamento de parâmetros equivalente. Isola quanto do ganho vem
de capacidade neural e não de estrutura relacional.
_Evitar_: GRU (nomeia a arquitetura, não o papel que o modelo cumpre no argumento).

## Avaliação

**Modo 1**:
Reconstrução da curva observada, dentro da amostra. Mede aderência, não previsão.
_Evitar_: fit retroativo, ajuste.

**Modo 2**:
Previsão causal a k semanas à frente, usando apenas informação anterior à origem.
É o modo que sustenta qualquer afirmação de superioridade preditiva.

**Origem de previsão**:
O par (música, chart) numa semana específica a partir da qual uma previsão é
emitida. É a unidade de observação do Modo 2. Uma música contribui com muitas
origens, que não são independentes entre si.
_Evitar_: música (confundir origem com música é o que inflou a significância no
texto de qualificação).

**Horizonte**:
O número k de semanas à frente da origem de previsão.
