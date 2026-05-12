# Plano de Pesquisa: Difusão Musical com Temporal GNN Heterogêneo

Replicação e extensão de:

> Oliveira, G. P.; Vassio, L.; Couto da Silva, A. P.; Moro, M. M.
> **Modeling music popularity as an epidemic: insights from the Brazilian market.**
> BraSNAM 2025. DOI: [10.5753/brasnam.2025.8760](https://doi.org/10.5753/brasnam.2025.8760)

Workshop alvo: **BraSNAM 2026** (mesma comunidade do paper original).

---

## 1. Status do dataset

Resolvido. Fonte única MGD+, dataset completo:

| Fonte | O que fornece | Cobertura BR |
|---|---|---|
| MGD+ Zenodo `8086643` + Viral 50 cedido | Top 200 + Viral 50 diários, features acústicas, gêneros, redes | 2017-01-01 → 2022-03-13 (1.895 dias) — **completo** |

**Números após cruzamento:**
- Top 200 BR: 200 entradas/dia, 2017-01-01 → 2022-03-13
- Viral 50 BR: 50 entradas/dia, 2017-01-01 → 2022-03-13
- Interseção viral∩hit: **1.981 músicas** (paper original: 1.977 — diferença mínima de deduplicação)
- 1.701 artistas, 530 gêneros distintos, rede gênero↔gênero com pesos pré-calculados

---

## 2. Por que o trabalho continua original (e o que mudou)

Os mesmos autores publicaram um follow-up em 2025 que precisa ser tratado com cuidado:

> Oliveira, G. P.; Vassio, L.; Couto da Silva, A. P.; Moro, M. M.
> **Contagious Rhythms: A Wave-Based Epidemic Approach for Music Virality on Social Platforms.**
> ASONAM 2025 / Springer LNCS 16322. DOI: [10.1007/978-3-032-13513-1_16](https://doi.org/10.1007/978-3-032-13513-1_16)

O que esse paper faz:
- Estende o SIR clássico com uma **abordagem wave-based**: múltiplas ondas SIR sobrepostas, capturando re-emergência (caso "Shallow").
- Avalia em 1.000+ músicas do Spotify.
- Compara com **algoritmos clássicos de previsão de séries temporais** (provavelmente ARIMA, Prophet, ETS).
- Conclui: wave-based é melhor para virality; long-term success continua difícil; forecast comparável a baselines de séries temporais com a vantagem de parâmetros interpretáveis.

**Implicações para o nosso paper:**

❌ **Não vender como contribuição:** "comparação com séries temporais" — Oliveira ASONAM 2025 já fez.

✅ **Posicionamento certo:** todos os trabalhos da linha do Oliveira et al. usam exclusivamente **modelos populacionais / agregados em série temporal**. Tratam cada música isoladamente, ignorando estrutura relacional. Nenhum deles incorpora informação de **artistas, gêneros, colaborações ou contexto da rede musical**.

**Tese deste trabalho:** modelos populacionais ignoram um sinal estrutural massivo. Um Temporal GNN heterogêneo que aprende sobre o grafo artista–música–gênero (a) reduz o erro de fit em hits de longa duração (onde o SIR é fraco — RMSE 2× maior que virality no paper original), (b) prevê melhor com horizonte k>1, (c) mantém parâmetros estruturalmente interpretáveis (atenção por tipo de aresta, importância de features de gênero/artista).

---

## 3. Pergunta de pesquisa e contribuições

**Pergunta:** sinais relacionais (colaborações entre artistas, vizinhança de gênero, co-trajetória entre músicas) capturam variância da popularidade musical brasileira que os modelos populacionais (SIR clássico) deixam de fora?

**Contribuições do paper:**

1. **Reformulação** do problema de difusão musical do MGD+ como aprendizado em grafo temporal heterogêneo. Schema explícito artista–música–gênero com co-trajetória direcionada.
2. **Comparação justa** contra o baseline populacional SIR (Oliveira BraSNAM 2025) em dois regimes (fit retroativo + predição genuína), com o mesmo dataset, pré-processamento e métricas.
3. **Análise interpretativa**: pesos de atenção por tipo de aresta, features mais informativas, casos de falha do GNN.

---

## 4. Plano por fases (10 semanas)

### Fase 0 — Reprodução dos baselines (semanas 1–2)

Antes de qualquer coisa, reproduzir os números do paper para garantir que sua pipeline está correta. Sem isso, qualquer comparação posterior é contaminada.

**Tarefas:**
- ✅ Pré-processamento: rank score, MA-7d, min-max [0, 0.5], floor 0.001.
- ✅ SIR fit com `scipy.integrate.odeint` + `scipy.optimize.curve_fit`. Initial guess β=γ=0.5, bounds [0, 10].
- ✅ Pipeline paralelo `joblib.Parallel(n_jobs=-1)`.

**Resultados (2026-05-12, 1.981 músicas, dataset completo):**
- Subset ≥ 1.900: ✅ 1.981
- Convergência ≥ 99%: ✅ 100%
- Mann-Whitney p ≈ 1e-60: ✅ 4,52e-125
- RMSE virality ≈ 0,028 ± 10%: ⚠️ 0,0381 (investigar se paper reporta mediana, não média)
- RMSE success ≈ 0,052 ± 10%: ⚠️ 0,0699 (idem)

### Fase 1 — Construção do grafo heterogêneo (semanas 2–3)

**Schema:**

| Tipo de nó | Quantidade | Features iniciais |
|---|---|---|
| Música | 6.469 | 9 acústicas + popularity + explicit + song_type + total_streams + dias_no_chart |
| Artista | 1.701 | num_hits + num_collab_hits + #anos no chart + #gêneros |
| Gênero | 530 | one-hot ou embedding aprendido |

**Tipos de aresta:**

| Aresta | Sentido | Origem | Features |
|---|---|---|---|
| `artista → música` (interpreta) | direcionada | listas `artist_ids` em songs | papel (principal/feat), índice na lista |
| `artista — gênero` (pertence_a) | não-direcionada | listas `genres` em artists | — |
| `música → música` (co-trajetória) | direcionada (ordem de entrada no chart) | charts | #dias_juntos, distância média de posição |
| `gênero — gênero` (co-ocorrência) | não-direcionada | genre_network MGD+ | weight, avg_popularity, avg_streams |

**Decisões importantes:**
- Co-trajetória: aresta entre músicas i, j se ambas estão no Top 200 (ou Viral 50) por ≥7 dias simultaneamente. Threshold evita arestas esparsas.
- Sub-grafos separados para virality e success (co-trajetória usa Viral 50 ou Top 200 conforme o experimento).
- Snapshots semanais para o componente temporal.

**Ferramentas:** `networkx` para construção e exploração estatística (distribuição de grau, componentes, clustering); `torch_geometric.data.HeteroData` para o pipeline de treino.

### Fase 2 — Modelagem com Temporal GNN heterogêneo (semanas 3–6)

**Arquitetura:** snapshot temporal + encoder heterogêneo + agregador recorrente.

```
[snapshot semana t] → HeteroGraphSAGE (2 camadas, hidden=128)
                    → embedding por música em t  (1 vetor 128-d)
                                ↓
[sequência de embeddings semana t-W..t-1] → GRU (hidden=128, 1 camada)
                                ↓
                              MLP → rank_score(t)  [escalar em [0, 0.5]]
```

Por que essa arquitetura para um workshop short paper:
- **HeteroGraphSAGE** é mais simples que HGT/HAN, mais fácil de defender e reproduzir; o paper se beneficia de simplicidade.
- **Snapshots semanais** evitam o overhead de TGN puro; perde resolução fina mas é mais robusto.
- **GRU sobre embeddings** é leve e captura dinâmica temporal sem o custo de Transformer temporal.
- **Total de parâmetros estimado**: ~200K — dá para treinar em CPU/laptop em horas.

**Alternativas se o resultado base não for bom:**
1. Substituir HeteroGraphSAGE por **HGT** (Heterogeneous Graph Transformer, Hu et al. 2020).
2. Substituir GRU por **Transformer encoder temporal**.
3. Migrar para **TGN** (Rossi et al. 2020) — TGN puro com grafo heterogêneo é caro mas mais expressivo.

**Hiperparâmetros do grid (pequeno, para caber no workshop):**
- Janela temporal W ∈ {4, 8, 12} semanas.
- Hidden dim ∈ {64, 128}.
- Layers GNN ∈ {2, 3}.
- Learning rate ∈ {1e-3, 5e-4} com Adam.

**Treino:** Loss = MSE no rank score normalizado. Early stopping em RMSE de validação (paciência=10 épocas).

**Splits temporais:**
- Treino: 2017-01 → 2020-06 (3,5 anos).
- Validação: 2020-07 → 2020-12 (6 meses).
- Teste: 2021-01 → 2021-12 (12 meses).

Respeita ordem temporal, evita data leakage. Validação é só para early stopping; teste é só para reportar números finais.

### Fase 3 — Avaliação dupla (semanas 6–8)

**Modo 1 — Fit retroativo (comparação 1-pra-1 com o paper original):**

Para cada música no conjunto de teste, o GNN gera a curva ajustada usando todos os dados disponíveis daquela música até o final do período. Mesmo regime do SIR no paper original. Métricas:
- RMSE médio ± IC 95%, separado por virality e success.
- Distribuição de RMSE (boxplot replicando Fig. 3 do paper).
- Mann-Whitney U comparando GNN vs SIR e GNN vs wave-based, par a par.

**Modo 2 — Predição genuína (extensão original):**

Para cada música e cada t ∈ teste, prediz rank_score em t+k usando apenas dados ≤ t. **Refazer o SIR e o wave-based no mesmo regime** para comparação justa: ajustar com dados ≤ t, projetar para t+k.

Métricas:
- RMSE de predição em horizontes k ∈ {1, 7, 14, 30 dias}.
- Acerto direcional (subiu/desceu).
- Score-CRPS (Continuous Ranked Probability Score) se conseguir extrair distribuição preditiva do GNN.

**Análise qualitativa de falhas do SIR (replicar Figs. 8 e 9 do paper):**
- "Shallow" (Lady Gaga & Bradley Cooper) — caso de re-emergência.
- "Batom de Cereja - Ao Vivo" (Israel & Rodolffo) — viral que vira hit longo.
- "Água Nos Zói - Ao Vivo" — hit de longa duração, RMSE mais alto do paper.
- "abcdefu" (GAYLE) — outro hit longo problemático.

Esperamos que o GNN acerte os casos de re-emergência e os hits longos onde o SIR clássico falha. O wave-based também acerta re-emergência, então a vantagem do GNN nesses casos é menor — mas em hits longos o GNN pode dominar porque usa contexto estrutural.

**Análise interpretativa (extra para diferencial do paper):**
- Atenção por tipo de aresta: que sinal o modelo usa mais (artista? gênero? co-trajetória?).
- Importância de features acústicas vs metadados.
- Análogos populacionais: extrair do GNN quantidades análogas a β (slope inicial), γ (slope de queda), R₀ (β/γ) para reportar nas Figs. 4–7 do paper.

### Fase 4 — Escrita e submissão (semanas 8–10)

**Estrutura sugerida** (8–12 páginas SBC):

1. **Introdução** — problema, gap (modelos populacionais ignoram estrutura), contribuição em 3 bullets.
2. **Trabalhos relacionados** — Hit Song Science (Seufitelli 2023a), modelos epidemiológicos em mídia/música (Rosati 2021, Sachak-Patwa 2018), citação obrigatória de Oliveira 2025 (BraSNAM), Oliveira 2025 (ASONAM, wave-based), Oliveira 2025 (IEEE Access, causalidade), GNNs temporais (Rossi 2020, Hamilton 2017, Hu 2020).
3. **Dados e pré-processamento** — descrever Kaggle + MGD+, justificar interseção, declarar limitação do período.
4. **Metodologia** — schema do grafo heterogêneo + arquitetura GNN; ser explícito sobre a tarefa de regressão e o protocolo de splits.
5. **Avaliação** — Modo 1, Modo 2, análise qualitativa, análise interpretativa.
6. **Discussão** — onde o GNN ganha, onde não ganha, custo computacional, casos de falha.
7. **Conclusão e trabalho futuro** — apontar para sua thesis (Proposta 1: temporal GNN para short-form video).

**Citações imprescindíveis (que não estavam no plano anterior):**
- Wave-based ASONAM 2025 (Oliveira et al.) — comparação direta.
- Causalidade IEEE Access 2025 (Oliveira et al.) — contexto da linha de pesquisa.
- WebSci 2024 (Oliveira et al., What makes a viral song) — features acústicas e contexto, justifica seu uso.
- Hu et al. 2020 (HGT) — alternativa de arquitetura.
- Rossi et al. 2020 (TGN) — alternativa de arquitetura.

---

## 5. Stack técnica

```
pandas, numpy             # manipulação
scipy                     # fit SIR (baseline)
networkx                  # construção e análise do grafo
torch, torch-geometric    # HeteroData, HeteroConv, SAGEConv
pytorch-lightning         # loop de treino limpo
wandb                     # tracking de experimentos
seaborn, matplotlib       # replicar figuras
```

Hardware: laptop com GPU é suficiente. Treino full em ~200K parâmetros sobre 1.981 músicas × 1.895 dias é da ordem de horas, não dias.

---

## 6. Critérios de sucesso (definir antes para evitar viés)

**Mínimo aceitável para submeter:**
- GNN bate SIR clássico em RMSE de success (Modo 1) com p < 0,01 no Mann-Whitney.
- GNN bate SIR clássico em pelo menos 2 dos 4 horizontes do Modo 2.

**Resultado forte:**
- GNN reduz RMSE em ≥30% nos casos de hits longos (subset com duração > 90 dias no chart).
- Análise interpretativa mostra atenção significativa em arestas artista→música e gênero↔gênero.

**Resultado limitado mas publicável:**
- GNN não bate SIR em todos os charts, mas mostra ganhos em subgrupo específico (por gênero, por duração).
- Reposicionar como "GNN é competitivo com modelos populacionais em uma fração relevante das músicas, com a vantagem de incorporar sinais relacionais".

---

## 7. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Overfitting do GNN com 1.981 músicas | Dropout alto, weight decay, early stopping agressivo, augmentation por subsampling de arestas |
| Co-trajetória pode introduzir vazamento (música A no chart "prediz" música B) | Validar splits temporais com cuidado; aresta só conta se data ≤ t no input |
| GNN não bater SIR em RMSE médio | Reposicionar como em "resultado limitado mas publicável" acima |
| Features acústicas dominarem o sinal estrutural | Ablation: GNN sem features acústicas vs com — se a estrutura não ajuda, é um resultado negativo honesto |

---

## 8. Próximos passos imediatos

1. Rodar `load_and_verify.py` localmente para confirmar carga.
2. Implementar Fase 0 (replicar SIR baseline) — sem isso, nada do resto vale.
3. Ler o paper ASONAM 2025 ("Contagious Rhythms") na íntegra para implementar fielmente o wave-based.
4. Estatísticas exploratórias do grafo heterogêneo construído antes de qualquer treino: distribuição de grau por tipo de nó, densidade, componentes conexas, comunidades por gênero.
5. Pipeline mínimo end-to-end com 100 músicas e arquitetura simples (HeteroSAGE 1 camada + GRU 1 camada) para validar que o sinal existe antes de escalar.

---

## 9. Referências do plano

- Oliveira et al. 2025 (BraSNAM) — paper original que estamos estendendo.
- Oliveira et al. 2025 (IEEE Access) — causalidade virality↔success, contextualização.
- Seufitelli et al. 2023b (DSW) — paper do MGD+, citação do dataset.
- Rossi et al. 2020 — TGN.
- Hu et al. 2020 — HGT.
- Hamilton et al. 2017 — GraphSAGE.
- Kempe et al. 2003 — IC e LT (útil para contextualizar diferença entre modelos populacionais e modelos de difusão em rede).
