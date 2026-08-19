# Segundo split, inteiramente pré-pandemia, como checagem de robustez

**Status:** aceito (2026-08-18)

O split usado na qualificação treina até **2020-06-29**, valida até 2020-12-21 e
testa de **2020-12-28 a 2022-03-13**. O treino é portanto quase todo pré-pandemia e
o teste é inteiramente pandemia e pós-pandemia. Isso é deslocamento de distribuição
entre treino e teste, não um detalhe de calendário, e favorece sistematicamente a
persistência: sob regime desconhecido, repetir o último valor é robusto, enquanto
qualquer modelo que aprendeu dinâmica pré-pandemia é penalizado. Pode explicar
parte do fato de a GNN só superar a persistência em k=4 no Top 200.

Acrescentamos um segundo regime de avaliação inteiramente pré-pandemia (treino
2017–2018, validação primeiro semestre de 2019, teste segundo semestre de 2019),
mantendo o split atual como principal. Se a ordenação entre persistência, SIR,
sem-grafo, GNN embaralhada e GNN completa se preservar nos dois regimes, o
confundidor está respondido com evidência em vez de parágrafo.

**Alternativa rejeitada:** adotar o split pré-pandemia como principal. Eliminaria o
confundidor de vez, mas descartaria cerca de 40% da janela temporal e invalidaria
todas as tabelas já escritas.

**Consequência:** o número de treinos dobra. Combinado com as três seeds, a matriz
fica em 18 treinos de modelo neural.
