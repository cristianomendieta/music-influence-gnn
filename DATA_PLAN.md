# Plano de Dados — Music Diffusion GNN

Fonte única, dataset completo, sem limitações de cobertura ou período.

---

## 1. Fontes de dados

| Dado | Fonte | Localização | Status |
|---|---|---|---|
| Top 200 BR diário | **MGD+** (Zenodo 8086643) | `data/charts/mgdplus/spotify_charts_regional_br.csv` | ✅ Completo — 200/dia |
| Viral 50 BR diário | **MGD+** (cedido pelo autor) | `data/charts/mgdplus/spotify_charts_viral_br.csv` | ✅ Completo — 50/dia |
| Features acústicas | **MGD+** | `data/songs/br-hit_songs-YYYY.csv` | ✅ Completo |
| Metadata artistas | **MGD+** | `data/artists/br-artists-all_time.csv` | ✅ Completo |
| Rede de gêneros | **MGD+** | `data/genre_network/br-genre_network-YYYY.csv` | ✅ Completo |

Todos os 68 mercados do MGD+ estão preservados em `data/MGDplus/` para futuro experimento cross-market.

---

## 2. Charts BR — especificações

| Métrica | Top 200 | Viral 50 |
|---|---|---|
| Entradas/dia | 200 | 50 |
| Período | 2017-01-01 → 2022-03-13 | 2017-01-01 → 2022-03-13 |
| Dias únicos | 1.895 | 1.890 |
| Dedup necessário | 2021-09-01 duplicado no top200 | — |

---

## 3. Subset viral∩hit

- **1.981 músicas** na interseção viral∩hit (vs 1.977 no paper — diferença de deduplicação).
- 100% das músicas do subset têm features acústicas no MGD+.
- Período das séries: 2017-01-01 → 2022-03-13.

---

## 4. Loader

`load_charts()` em `src/music_diffusion_gnn/data/loaders.py` lê todos os `*.csv` em `data/charts/mgdplus/` e normaliza `Chart` → `{top200, viral50}`. Erro explícito se a pasta não existir.

---

## 5. Pré-processamento (idêntico ao paper)

```
rank_score = max_rank − rank + 1
    ↓ média móvel 7 dias (min_periods=1)
    ↓ min-max → [0, 0.5]
    ↓ floor: dias ausentes → 0.001
```

---

*Atualizado em: 2026-05-12 — dataset completo obtido, Kaggle descartado.*
