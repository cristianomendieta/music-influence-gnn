# Experimentos — Phase 0: Reprodução do Baseline SIR

> Documento de registro dos experimentos realizados.

---

## 1. Fonte de Dados

| Atributo | Valor |
|---|---|
| Charts | Spotify Top 200 BR + Viral 50 BR |
| Fonte | **MGD+** (Zenodo 8086643) + Viral 50 cedido pelo autor |
| Localização | `data/charts/mgdplus/` |
| Período | 2017-01-01 → 2022-03-13 |
| Total de dias | ~1.895 (top200) / ~1.890 (viral50) |
| Músicas na interseção viral∩hit | **1.981** (paper: 1.977) |
| Cobertura top200 | 200 entradas/dia ✅ |
| Cobertura viral50 | 50 entradas/dia ✅ |

---

## 2. Pré-processamento

Pipeline idêntico ao paper (Seção 4.1):

```
rank_score = max_rank − rank + 1
    ↓ média móvel 7 dias (rolling, min_periods=1)
    ↓ normalização min-max → [0, 0.5]
    ↓ floor: dias ausentes → 0.001
```

- **Top 200:** max_rank = 200
- **Viral 50:** max_rank = 50

Cada música gera **duas séries independentes** (~1.895 dias): `viral50` (viralidade) e `top200` (sucesso).

---

## 3. Baseline SIR Clássico

### 3.1 Implementação

- **EDO:** `dS/dt = −βSI`, `dI/dt = βSI − γI`, `dR/dt = γI`
- **Condições iniciais:** `I(0) = y[0]`, `S(0) = 1 − y[0]`, `R(0) = 0`
- **Solver:** `scipy.integrate.odeint`
- **Otimizador:** `scipy.optimize.curve_fit`, initial guess `[β, γ] = [0.5, 0.5]`, bounds `[0, 10]`
- **Paralelização:** `joblib.Parallel(n_jobs=-1)`

### 3.2 Resultados (MGD+ completo, 1.981 músicas)

Executado em 2026-05-12 via `python scripts/run_phase0.py --force`.

| Métrica | Target (paper) | Tolerância | **Obtido** | Status |
|---|---|---|---|---|
| RMSE médio — virality (viral50) | ≈ 0,028 | ± 10% | **0,0381** | ⚠️ +36% |
| RMSE médio — success (top200) | ≈ 0,052 | ± 10% | **0,0699** | ⚠️ +34% |
| Convergência do otimizador | alta | ≥ 99% | **100%** | ✅ |
| Mann-Whitney p (success vs virality) | ordem 1e-60 | mesma ordem | **4,52e-125** | ✅ |
| Subset size | 1.977 | ≥ 1.900 | **1.981** | ✅ |

### 3.3 Análise da discrepância no RMSE

Com dataset completo (1.981 songs), o RMSE médio continua acima do target. A mediana provavelmente está mais próxima. Hipóteses:

- O paper pode reportar **mediana**, não média (a verificar na Seção 4.2 do paper).
- Músicas com padrão multi-onda (longa duração) têm RMSE muito alto e elevam a média.
- Possível diferença nos bounds do `curve_fit` ou na condição inicial.

**Ação:** verificar se o paper usa média ou mediana antes de investigar o fitting. Se for mediana, os resultados estão dentro da tolerância.

---

## 4. Artefatos Gerados

| Arquivo | Descrição |
|---|---|
| `data/processed/subset_ids.json` | 1.981 song_ids do subset viral∩hit |
| `data/processed/timeseries.parquet` | Séries temporais normalizadas (7,5M linhas) |
| `results/phase0/sir_params.parquet` | β, γ, R₀, RMSE, convergência por música e chart |
| `results/phase0/summary.md` | Tabela R4 com status dos critérios |
| `results/phase0/boxplot_fig3.png` | Réplica da Fig. 3 do paper |

---

*Atualizado em: 2026-05-12 — dataset completo, wave-based descartado por decisão do pesquisador.*
