# 16 — Documento: fórmula da cabeça temporal e descrições desatualizadas

**What to build:** o capítulo de metodologia passa a descrever a cabeça temporal que o código de fato executa, e as demais divergências texto ↔ código ficam explicitamente resolvidas ou registradas como decisão.

A divergência principal: `documento_qualificao/4-metodologia/texto.tex:125` afirma `ŷ_i = 0,5 · σ(MLP(h_i))`, a formulação anterior à revisão R1. O código implementa cabeça **residual**: `Δ = MLP(h)` e `ŷ = clamp(y_prev + Δ, 0, 0,5)`, com `y_prev = pop_bank[w−1, song, chart]`, que é exatamente o baseline de persistência ingênua (`diffusion_gnn.py:184-193`). Os slides já foram corrigidos em 12/08; o documento não. É a divergência mais grave que sobrou, porque um arguidor que leia o documento e o código vê duas arquiteturas diferentes. Trocar a fórmula obriga a mencionar a âncora, o que também antecipa a suspeita de vazamento: a âncora é sempre a semana anterior, informação passada.

Duas decisões conscientes precisam ficar registradas no mesmo passe, para que nenhuma sessão futura as "conserte" de novo: a imputação continua descrita como **mediana** no texto e nos slides embora o código impute zero no espaço padronizado (a média), e o bloco de normalização continua dizendo que os limites são teóricos e fixos embora `preprocess.py:29-31` use `smoothed.min()/max()` por série. As duas foram mantidas por decisão explícita em 11 e 12/08. O que falta é a **ressalva no texto** para a segunda, já que hoje ela é usada como argumento de ausência de vazamento no alvo.

Contexto completo em `docs/achados-qualificacao.md`, itens A1, A2 e A3.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] §4.7 descreve a cabeça residual com `clamp` e a âncora `y_prev`, não a sigmoide escalada
- [ ] O texto diz explicitamente que a âncora é a mesma informação do baseline de persistência
- [ ] A normalização min-max ganha a ressalva de que usa o máximo observado da própria série, e a limitação é declarada como igual para todos os modelos comparados
- [ ] A escolha de manter "mediana" fica registrada como decisão, com a resposta pronta, e não como pendência
