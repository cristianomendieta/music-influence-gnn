# 19 — Cabeça temporal cega ao chart

**What to build:** a correção estrutural prevista pelo modelo passa a poder diferir entre Viral 50 e Top 200.

Hoje a sequência que entra na GRU não tem dimensão de chart. O código do chart é usado **uma vez só**, para buscar a âncora `y_prev = pop_bank[w−1, song, chart]` (`diffusion_gnn.py:190-193`). Para a mesma música e a mesma semana, portanto, **o Δ do Viral 50 e o do Top 200 é literalmente o mesmo número**, e as duas previsões diferem apenas pelo ponto de partida.

Não é inócuo. Os dois canais de popularidade entram nos atributos do nó, então o vetor da música **sabe** dos dois charts, mas o modelo não consegue aprender que a estrutura empurra a viralidade para um lado e o volume consolidado para outro, que é justamente o tipo de assimetria que o trabalho argumenta existir entre os dois regimes. É visível em cinco linhas de código, então é melhor endereçar do que esperar ser perguntado.

Duas variações possíveis, a decidir na implementação: separar a cabeça por chart, ou passar o código do chart como entrada da cabeça (embedding de duas linhas concatenado a cada passo da sequência). A segunda é mais barata e mantém um modelo só.

Contexto em `docs/achados-qualificacao.md`, item C2.

**Blocked by:** 06 (comparação contra a GNN re-treinada na matriz nova)

**Status:** ready-for-agent

- [ ] O Δ previsto pode diferir entre os dois charts para a mesma música e semana
- [ ] Comparação célula a célula contra a cabeça atual, mesmo protocolo e mesmas seeds
- [ ] Se a variação não melhorar, o resultado negativo é reportado e a cegueira ao chart vira limitação declarada em vez de defeito silencioso
