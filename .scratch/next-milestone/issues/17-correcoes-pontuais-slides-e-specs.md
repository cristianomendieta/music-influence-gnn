# 17 — Correções pontuais em slides e specs

**What to build:** três correções de uma linha cada, todas apuradas na preparação da qualificação e nenhuma com efeito em resultado. Estão juntas porque separadas não pagam um ticket.

Primeira, `.specs/features/phase-1-hetero-graph/design.md:17` diz que a aresta reversa vem do `ToUndirected()` do PyG; o build a cria à mão com `flip(0)` (`build.py:141`). O resultado é o mesmo e o `docs.md` já descreve corretamente, é só a linha do design que ficou para trás.

Segunda, o slide 26 afirma "tabela 530×32 otimizada junto com os pesos: **a rede descobre a geometria**". A primeira metade é verdadeira, a segunda a medição contradiz: no checkpoint avaliado a tabela colapsou (514 das 530 linhas com norma < 0,001, ver `docs/achados-qualificacao.md` B4). A frase precisa descrever a **decisão de projeto** — identidade aprendida, por não haver atributo que descreva um gênero — e não o resultado. O item 05 pode tornar a frase obsoleta ao substituir a tabela por atributos estruturais; se isso acontecer primeiro, a correção vira reescrita.

Terceira, o diagrama de pré-processamento do slide 20 mostra seis etapas e são **sete**: falta a densificação, que cria um calendário diário contínuo do lançamento da música até o fim da janela e preenche todo dia fora do chart com **zero**, não com ausente (`preprocess.py:89-93`). Não é detalhe de implementação: é a decisão que produz o fato de ~95% dos alvos do teste serem o valor de piso, hoje declarado na introdução pelo item 13. O diagrama esconde a origem do que o texto declara.

Contexto em `docs/achados-qualificacao.md`, itens A5, A6 e A7.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `design.md` da Fase 1 descreve a construção real da reversa
- [ ] O bloco do slide 26 descreve a decisão de projeto, não um resultado que a medição não sustenta
- [ ] A densificação aparece no pipeline do slide 20, e a ligação com a composição do teste fica explícita
