# Validação — Phase 6b, mapa de onde a estrutura paga

## Validation: phase-6b-mapa-estrutura - PASS (código e testes)

**Result**: PASS para o que roda localmente (funções puras, testes, notebook sem execução). O mapa em si ainda não foi calculado: falta rodar `notebooks/item22_mapa_estrutura_colab.ipynb` no Colab.

**Limite desta validação.** Foi feita pelo mesmo agente que implementou, no modo *standalone fallback* da skill (sem sub-agente, a sessão não autorizou spawn). Não cumpre "autor ≠ verificador". Os mutantes abaixo foram desenhados depois de ver os testes, então a contagem de mortos não substitui uma revisão independente. Recomendo uma segunda leitura (`/code-review`) antes de citar números do Colab.

**Diff range:** `53e6b47..9f4d8eb` mais a correção do teste de vazamento desta validação (commit seguinte). Gate: `pytest tests/test_strata.py tests/test_mapa.py` → 57 passed; `ruff check` limpo; `scripts/check_notebook.py` → 0 erros.

## Verificação ancorada na spec (AC → evidência)

| AC | Evidência (`file:line` + asserção) | Resultado esperado pela spec | Cobre? |
|---|---|---|---|
| MAP-01 AC1: cada origem recebe exatamente um dos cinco estratos | `tests/test_strata.py:91` — `set(out.unique()) <= set(STRATA)` e `len(out) == len(origins)`; `tests/test_strata.py:103` — `list(out) == ["estreia","jovem","reentrada","maduro_multi"]` | um estrato por origem, os cinco existem | ✅ |
| MAP-01: limites de idade e de hiato | `tests/test_strata.py:40` — idade 8 `jovem`, 9 `maduro_mono`; `tests/test_strata.py:57` e `:63` — hiato 3 uma onda, hiato 4 duas | H = 8, hiato = 4 | ✅ |
| MAP-02 AC2: só semanas ≤ origem | `tests/test_strata.py:69` — `_stratum(oc,"a","top200",14) == "reentrada"` com onda futura; `:129` — mono em `w=25`, multi em `w=38`; `:136` — série futura não muda origens anteriores | passado apenas | ✅ |
| MAP-02 AC3: truncar a série em `w` não muda o estrato | `tests/test_strata.py:119` — `on_cut.iloc[0] == on_full[...].iloc[0]`, 4 séries, 50 origens cada | invariância | ✅ |
| MAP-01 AC4: origem fora da série é `sem_serie` | `tests/test_strata.py:152` — `list(out) == ["maduro_mono","sem_serie","estreia"]` | `sem_serie` e excluída dos testes | ✅ (exclusão: `tests/test_mapa.py:283`) |
| MAP-03 AC5: contagens e corte de 30 | `tests/test_strata.py:184` — 29 músicas `insuficiente`, 30 não; `:198` — soma das origens = total | corte `< 30` | ✅ |
| MAP-04 AC1: diferença, IC, `p` por música | `tests/test_mapa.py:33` — `mean_diff == -0.2`, `win_rate == 1.0`, `n_songs == 3`; `:44` — IC e `p` iguais a `paired_by_song`; `:62` — música com 2 origens é 1 observação | IC 95% bootstrap, Wilcoxon por música | ✅ |
| MAP-04: diferença nula não levanta | `tests/test_mapa.py:72` — `p_value == 1.0`, `(ci_lo, ci_hi) == (0.0, 0.0)` | edge case da spec | ✅ |
| MAP-05 AC2: uma família de Holm por split, dois contrastes | `tests/test_mapa.py:133` e `:148` — `holm_family(...) == holm_correction(p)`; `:261` — na tabela | 1 família/split, ≤ 30 | ✅ |
| MAP-05 AC3: rótulos | `tests/test_mapa.py:89`–`:121` — vence, perde, equivalente, inconclusivo, limites `p_holm = 0,05`, IC tocando zero, IC na margem, precedência | regra da spec | ✅ |
| MAP-06 AC4: estrutura paga | `tests/test_mapa.py:202` — `verdict["estreia"] == "estrutura paga"`; `:212` um split só → `nao paga`; `:220` um controle só → `nao paga`; `:228` célula pequena → `nao testavel` | vence os dois, ≥ 2 de 3, nos dois splits | ✅ (leitura estrita, ver lacuna 1) |
| MAP-04/05: uma linha por (split, estrato, k, controle), rótulo recomputável | `tests/test_mapa.py:237` — `len(table) == 2*2*3*2`; `:247` — `r.rotulo == label_cell(r.ci_lo, r.ci_hi, r.p_holm, r.margin)` | rastreável | ✅ |
| MAP-07 AC1–3: seed, chart, margem | `tests/test_mapa.py:306` — `seeds_concordam == 3` e `== 2`; `:325` — `dif_rmse == 0.1 - 0.2` e sem `p`/rótulo; `:362` — só a célula `k=1` muda | sensibilidade | ✅ |
| MAP-06 AC5–6: hipótese no cabeçalho, saídas em `results/mapa_estrutura/` e relatório | `notebooks/item22_mapa_estrutura_colab.ipynb` células 0 e 17 (inspeção); `scripts/check_notebook.py:19` garante que compila e não tem saída | hipótese antes dos resultados | ⚠️ inspeção, não asserção |
| MAP-08 (grau de cotrajetória) | — | P3, fora das tasks | adiado |

## Lacunas de precisão da spec

1. **"Vence os dois controles em ≥ 2 de 3 horizontes"** admitia duas leituras (o mesmo `k` nos dois controles, ou cada controle em ≥ 2 horizontes quaisquer). Foi fixada a estrita e registrada na spec (linha "Leitura estrita"). Os testes em `tests/test_mapa.py:220` e `:202` ancoram essa leitura.
2. **`nao testavel`**: a spec só cobria "todas as origens insuficientes". Foi estendido para "algum split com menos de 2 horizontes testáveis", também registrado na spec.

## Discriminação (sensor)

23 mutações de comportamento, aplicadas numa cópia temporária de `src/` e `tests/` (a árvore real não foi tocada), cada uma rodando a suíte inteira: **23 mortas, 0 sobreviventes.** Uma sobreviveu na primeira rodada (M3: contar ondas com a série inteira, vazamento do futuro): a série de teste não tinha uma origem madura anterior a uma segunda onda. Foi corrigido com `tests/test_strata.py:129` e o caso extra em `:113`, e a rodada repetida.

Mutações: limite de idade e de hiato, vazamento de ondas, semana da origem no passado, reentrada removida, corte de 30 com `<=`, `sem_serie` sem marcação, `win_rate` invertido, margem pelo tratado, diferença nula sem tratamento, Holm com `<=`, IC tocando zero como vitória, precedência equivalente-antes-de-vence, `insuficiente` dentro da família, família por contraste, recorte on-chart ignorado, `sem_serie` no mapa, vencer um controle basta, vencer em um split basta, `nao testavel` exigindo todos os splits, `by_chart` em MSE, `margin_sensitivity` vazia, concordância contando todas as seeds.

## O que não foi verificado

- **O notebook nunca rodou com os dados completos.** Um smoke com 150 músicas, `B = 100` e `MIN_SONGS = 3` percorreu todas as células de código sem erro (7 s) e escreveu os arquivos de saída num diretório temporário. Os números dessa rodada são ruído e não valem nada. Não foi exercitado o ramo de sensibilidade por seed (não há `mode2_*_completo.parquet` aqui).
- A asserção de que as origens somam 141.504 (`current`) e 53.982 (`pre_pandemia`) está no notebook e só roda no Colab.
- `sensitivity_by_seed` assume o esquema dos `mode2_*_completo.parquet` do item 09 (`variante`, `seed`, `onchart`, `split_regime`, `y_true`, `y_pred`), lido do código do notebook 09, não de um arquivo real.
