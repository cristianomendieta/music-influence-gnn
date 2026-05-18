# Phase 0 — Design

> Arquitetura, decisões e contratos. Requisitos formais em [`spec.md`](spec.md).

## Visão de alto nível

```
data/charts/spotify_charts_br_2017_2021.csv
data/songs/br-hit_songs-*.csv
data/artists/br-artists-all_time.csv
            │
            ▼
┌─────────────────────────┐
│ data.loaders            │  carrega CSVs, normaliza tipos, retorna dataframes
│ data.subset             │  identifica viral∩hit (1179 ids)
│ data.preprocess         │  rank score → MA-7d → min-max [0,0.5] → floor 0.001
└─────────────────────────┘
            │
            ▼  data/processed/timeseries.parquet
            │  data/processed/subset_ids.json
            ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│ baselines.sir           │    │ baselines.wave_based    │
│  fit_sir(series)        │    │  fit_wave(series, M*)   │
│  → β, γ, R0, rmse       │    │  → params, rmse         │
└─────────────────────────┘    └─────────────────────────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
                 ┌─────────────────────────┐
                 │ evaluation.metrics      │  RMSE, Mann-Whitney
                 │ evaluation.report       │  tabelas + boxplot Fig.3-like
                 └─────────────────────────┘
                           │
                           ▼
              results/phase0/{sir,wave}_params.parquet
              results/phase0/rmse_per_song.parquet
              results/phase0/summary.md
```

## Decisões de design

### D1 — Pré-processamento idempotente e cacheado

`data.preprocess.build_timeseries()` lê os CSVs crus uma vez e escreve
`data/processed/timeseries.parquet` com schema:

```
song_id (str) | chart (top200|viral50) | date (date) | rank_score (float) | y (float)
```

Onde `y` é a série final pós floor. Long format escolhido para facilitar
filtragens por subset e por chart sem reabrir CSVs gigantes.

**Por quê parquet:** Top 200 + Viral 50 × 1.826 dias = ~14M linhas.
Carregamento ~10× mais rápido que CSV; tamanho ~5× menor.

### D2 — Subset persistido como JSON

`data/processed/subset_ids.json` contém:
```json
{
  "viral_intersect_hit": ["<song_id>", ...],
  "n": 1179,
  "generated_at": "2026-05-XX"
}
```
Pequeno, versionável, evita recomputar interseção em cada experimento.

### D3 — SIR como módulo puro, sem dependência de pandas

`baselines.sir.fit_sir(y: np.ndarray, t: np.ndarray) -> SIRFit` recebe arrays
e retorna dataclass com β, γ, R₀, RMSE, flag `success` (do `curve_fit`).
Justificativa: facilita teste unitário com séries sintéticas e paraleliza fácil.

EDO clássica:
```
dS/dt = -β·S·I
dI/dt =  β·S·I - γ·I
dR/dt =  γ·I
```
A "população" de cada música é normalizada para `S(0) = 1 - I(0)`,
`I(0) = y[0]` (primeiro valor da série), `R(0) = 0`. O fit ajusta apenas β e γ
contra `I(t)` (a série observada).

### D4 — Wave-based como soma compositiva de SIR

`baselines.wave_based.fit_wave(y, t, M_max=5) -> WaveFit` testa M ∈ {1, ..., M_max}
e seleciona pelo BIC. Para cada M:
- Otimização inicial via `scipy.optimize.differential_evolution` (escapa
  mínimos locais), refino via `curve_fit`.
- Cada onda i tem 3 parâmetros: β_i, γ_i, t0_i (offset de início).
- Total: 3M parâmetros + 1 escala global → 3M + 1.

**BIC vs AIC:** BIC penaliza mais a complexidade. Com 1.826 pontos por série,
BIC tende a escolher M menor — alinhado com a interpretabilidade que é a
vantagem alegada do wave-based no paper. **Decidido: BIC.**

`M_max = 5` é generoso o bastante para casos "Shallow" (re-emergência) e
limita custo computacional (≤16 parâmetros por música).

### D5 — Fallback do wave-based registrado, não pré-implementado

Mistura de Gaussianas como fallback **só entra se** a reprodução fiel do
wave-based falhar nos critérios numéricos da R4 e a tentativa de contato
com o autor não destravar em ≤3 dias. Decisão registrada em STATE.md no
momento que ocorrer; não há código preventivo agora.

### D6 — Paralelização

1.179 músicas × 2 charts × 2 modelos = 4.716 fits independentes.
SIR é leve (segundos por música); wave-based é pesado (até minuto por música
com differential_evolution e M_max=5).

`baselines.parallel.fit_all(songs, model)` usa `joblib.Parallel` com
`n_jobs=-1` (todos os cores). Sem GPU nesta fase. Estimativa: SIR completo
em <5 min; wave-based completo em ~1–2h.

### D7 — Avaliação e reporting

`evaluation.metrics.rmse_pairwise(model_a, model_b) -> {p_value, statistic}`
roda Mann-Whitney pareado.

`evaluation.report.write_summary()` gera `results/phase0/summary.md` com:
- Tabela R4 preenchida.
- Boxplot RMSE virality vs success por modelo (replica Fig. 3).
- Top 10 piores fits do SIR (candidatos a "Shallow", "Batom de Cereja", etc).

## Module breakdown

```
src/music_diffusion_gnn/
├── data/
│   ├── loaders.py           # load_charts(), load_songs(), load_artists()
│   ├── subset.py            # build_subset() → JSON
│   └── preprocess.py        # build_timeseries() → parquet
├── baselines/
│   ├── sir.py               # SIRFit dataclass, fit_sir()
│   ├── wave_based.py        # WaveFit dataclass, fit_wave()
│   └── parallel.py          # fit_all() com joblib
├── evaluation/
│   ├── metrics.py           # rmse(), mann_whitney_pairwise()
│   └── report.py            # write_summary(), make_boxplot()
└── ...
```

```
scripts/
├── exploratory/verify_data.py   # diagnóstico/sanidade (não faz parte do pipeline)
└── run_phase0.py            # orquestra: subset → preprocess → fits → report
```

```
tests/                       # NOVO
├── test_preprocess.py       # rank score + MA-7d + min-max + floor em série sintética
├── test_sir.py              # fit em curva SIR sintética: recupera β, γ
└── test_wave_based.py       # fit em soma de 2 SIR sintéticos: recupera M=2
```

## Contratos / interfaces

### `data.preprocess.build_timeseries`

```python
def build_timeseries(
    raw_charts_path: Path,
    out_path: Path,
    *,
    floor: float = 0.001,
    window: int = 7,
    target_max: float = 0.5,
) -> pd.DataFrame:
    """Constrói matriz long-format e persiste em parquet."""
```

### `baselines.sir.fit_sir`

```python
@dataclass
class SIRFit:
    beta: float
    gamma: float
    R0: float
    rmse: float
    converged: bool
    n_iter: int

def fit_sir(y: np.ndarray, t: np.ndarray | None = None) -> SIRFit: ...
```

### `baselines.wave_based.fit_wave`

```python
@dataclass
class WaveFit:
    M: int
    waves: list[tuple[float, float, float]]  # (beta, gamma, t0) por onda
    bic: float
    rmse: float
    converged: bool

def fit_wave(y: np.ndarray, t: np.ndarray | None = None, M_max: int = 5) -> WaveFit: ...
```

### `baselines.parallel.fit_all`

```python
def fit_all(
    timeseries: pd.DataFrame,
    fit_fn: Callable[[np.ndarray], Any],
    *,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Aplica fit_fn em cada (song_id, chart). Retorna df indexado por (song_id, chart)."""
```

## Riscos específicos da fase

| Risco | Probabilidade | Mitigação |
|---|---|---|
| `curve_fit` falha em séries muito esparsas (poucos dias > floor) | média | Filtrar séries com <14 dias úteis antes do fit; reportar como N/A |
| `differential_evolution` lento demais em laptop | média | Cap em `maxiter=200`, `popsize=12`; revisar se exceder 2h total |
| RMSE alvo do paper foi reportado em subset com período até 2022-03 | confirmada | Tolerância ± 10%; declarar limitação |
| Numerical overflow em ODE com β grande | baixa | Bounds superiores explícitos (β, γ ≤ 10) |

## Open questions resolvidas neste design

- **AIC vs BIC?** → BIC (D4).
- **`curve_fit` para wave-based?** → não, `differential_evolution` + `curve_fit` (D4).
- **Escala temporal?** → dia, igual ao paper (R0 herdado da spec).

Nenhum gray area requer discuss; segue para tasks.
