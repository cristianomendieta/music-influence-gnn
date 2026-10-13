"""Unit tests for evaluation/strata.py (Phase 6b, MAP-01..MAP-03)."""
from __future__ import annotations

import pandas as pd
import pytest

from music_diffusion_gnn.evaluation.strata import STRATA, assign_strata, strata_counts


def _onchart(**series: list[int]) -> set[tuple[str, str, int]]:
    """``{"a|viral50": [10, 11]}``-style keyword helper → on-chart week set."""
    out = set()
    for key, weeks in series.items():
        song, chart = key.split("__")
        out |= {(song, chart, w) for w in weeks}
    return out


def _origins(song: str, chart: str, weeks: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"song_id": song, "chart": chart, "origin_week": weeks})


def _stratum(onchart, song, chart, week, **kw) -> str:
    return assign_strata(_origins(song, chart, [week]), onchart, **kw).iloc[0]


# ---------------------------------------------------------------- MAP-01: five strata

def test_origin_before_any_on_chart_week_is_estreia():
    oc = _onchart(a__viral50=list(range(10, 20)))
    assert _stratum(oc, "a", "viral50", 5) == "estreia"
    assert _stratum(oc, "a", "viral50", 9) == "estreia"


def test_origin_in_first_on_chart_week_is_jovem():
    oc = _onchart(a__viral50=list(range(10, 20)))
    assert _stratum(oc, "a", "viral50", 10) == "jovem"


def test_age_8_is_jovem_and_age_9_is_maduro():
    oc = _onchart(a__top200=list(range(10, 40)))
    assert _stratum(oc, "a", "top200", 18) == "jovem"         # 18 - 10 = 8
    assert _stratum(oc, "a", "top200", 19) == "maduro_mono"   # 19 - 10 = 9


def test_young_weeks_is_configurable():
    oc = _onchart(a__top200=list(range(10, 40)))
    assert _stratum(oc, "a", "top200", 15, young_weeks=4) == "maduro_mono"


def test_off_chart_origin_with_history_is_reentrada():
    oc = _onchart(a__viral50=[10, 11, 12])
    assert _stratum(oc, "a", "viral50", 13) == "reentrada"
    assert _stratum(oc, "a", "viral50", 40) == "reentrada"


def test_gap_of_three_weeks_keeps_one_wave():
    # on 10-12, off 13-15 (3 weeks), on 16-30
    oc = _onchart(a__top200=[10, 11, 12] + list(range(16, 31)))
    assert _stratum(oc, "a", "top200", 25) == "maduro_mono"


def test_gap_of_four_weeks_opens_a_second_wave():
    # on 10-12, off 13-16 (4 weeks), on 17-30
    oc = _onchart(a__top200=[10, 11, 12] + list(range(17, 31)))
    assert _stratum(oc, "a", "top200", 25) == "maduro_multi"


def test_wave_count_ignores_weeks_after_the_origin():
    # second wave starts at 17; an origin at 14 (off-chart) must not see it
    oc = _onchart(a__top200=[10, 11, 12] + list(range(17, 31)))
    assert _stratum(oc, "a", "top200", 14) == "reentrada"
    # origin inside the first wave, young → jovem regardless of the later wave
    assert _stratum(oc, "a", "top200", 12) == "jovem"


def test_wave_gap_is_configurable():
    oc = _onchart(a__top200=[10, 11, 12] + list(range(16, 31)))   # 3-week gap
    assert _stratum(oc, "a", "top200", 25, wave_gap=3) == "maduro_multi"


def test_charts_of_the_same_song_are_independent():
    oc = _onchart(a__viral50=list(range(10, 40)), a__top200=[30, 31])
    assert _stratum(oc, "a", "viral50", 35) == "maduro_mono"
    assert _stratum(oc, "a", "top200", 35) == "reentrada"
    assert _stratum(oc, "a", "top200", 20) == "estreia"


# ---------------------------------------------------- MAP-01: exclusive and exhaustive

def test_every_origin_gets_exactly_one_known_stratum_and_index_is_kept():
    oc = _onchart(a__viral50=list(range(10, 40)), b__top200=[5, 6, 20, 21])
    rows = [("a", "viral50", w) for w in range(0, 50, 3)] + [("b", "top200", w) for w in range(30)]
    origins = pd.DataFrame(rows, columns=["song_id", "chart", "origin_week"])
    origins.index = range(1000, 1000 + len(origins))
    out = assign_strata(origins, oc)
    assert len(out) == len(origins)
    assert list(out.index) == list(origins.index)
    assert set(out.unique()) <= set(STRATA)
    assert out.notna().all()


def test_all_five_strata_appear_in_a_series_built_to_cover_them():
    oc = _onchart(a__top200=[10, 11, 12] + list(range(17, 31)))
    out = assign_strata(_origins("a", "top200", [5, 10, 14, 25]), oc)
    assert list(out) == ["estreia", "jovem", "reentrada", "maduro_multi"]
    oc2 = _onchart(b__top200=list(range(10, 40)))
    assert _stratum(oc2, "b", "top200", 30) == "maduro_mono"


# ------------------------------------------------------------ MAP-02: no future leakage

@pytest.mark.parametrize("weeks", [
    list(range(10, 40)),
    [10, 11, 12] + list(range(17, 31)),
    [3, 4, 9, 10, 11, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
])
def test_stratum_is_invariant_to_truncating_the_series_at_the_origin(weeks):
    full = _onchart(a__top200=weeks)
    origins = _origins("a", "top200", list(range(50)))
    on_full = assign_strata(origins, full)
    for w in origins.origin_week:
        truncated = {(s, c, wk) for (s, c, wk) in full if wk <= w}
        on_cut = assign_strata(origins[origins.origin_week == w], truncated)
        assert on_cut.iloc[0] == on_full[origins.origin_week == w].iloc[0], f"w={w}"


def test_future_only_series_does_not_change_earlier_origins():
    base = _onchart(a__top200=[10, 11])
    extended = base | {("a", "top200", w) for w in range(50, 60)}
    origins = _origins("a", "top200", list(range(45)))
    assert list(assign_strata(origins, base)) == list(assign_strata(origins, extended))


# ----------------------------------------------------------------- edge cases (spec)

def test_origin_week_before_the_first_week_of_the_data_is_estreia():
    oc = _onchart(a__viral50=[0, 1, 2])
    assert _stratum(oc, "a", "viral50", -3) == "estreia"


# ------------------------------------------------- MAP-01 (AC4): origins without a series

def test_origin_outside_the_series_is_sem_serie_and_the_rest_is_untouched():
    oc = _onchart(a__top200=list(range(10, 40)))
    origins = pd.DataFrame({
        "song_id": ["a", "ghost", "a"], "chart": ["top200", "top200", "viral50"],
        "origin_week": [30, 30, 30],
    })
    out = assign_strata(origins, oc, series_keys={("a", "top200"), ("a", "viral50")})
    assert list(out) == ["maduro_mono", "sem_serie", "estreia"]


def test_series_keys_none_keeps_every_origin_in_the_five_strata():
    oc = _onchart(a__top200=[10])
    out = assign_strata(_origins("ghost", "top200", [30]), oc)
    assert out.iloc[0] == "estreia"


# ---------------------------------------------------- MAP-03: counts and the 30-song cut

def _cells(n_songs: int, n_origins: int, *, estrato="jovem", split="current", k=1, charts=("top200",)):
    rows = []
    for i in range(n_origins):
        rows.append({"split_regime": split, "estrato": estrato, "k": k,
                     "song_id": f"s{i % n_songs}", "chart": charts[i % len(charts)]})
    return pd.DataFrame(rows)


def test_counts_report_origins_and_distinct_songs():
    out = strata_counts(_cells(n_songs=10, n_origins=100))
    row = out.iloc[0]
    assert (row.n_origens, row.n_musicas) == (100, 10)


def test_cell_with_29_songs_is_insuficiente_and_30_is_not():
    assert strata_counts(_cells(29, 29)).iloc[0].insuficiente
    assert not strata_counts(_cells(30, 30)).iloc[0].insuficiente


def test_min_songs_is_configurable():
    assert not strata_counts(_cells(10, 10), min_songs=10).iloc[0].insuficiente


def test_a_song_in_both_charts_counts_once_in_n_musicas():
    out = strata_counts(_cells(n_songs=5, n_origins=10, charts=("viral50", "top200")))
    assert out.iloc[0].n_musicas == 5 and out.iloc[0].n_origens == 10


def test_counts_sum_back_to_the_input_origins_including_sem_serie():
    parts = [_cells(40, 80, estrato=e, k=k)
             for e in (*STRATA, "sem_serie") for k in (1, 2, 4)]
    df = pd.concat(parts, ignore_index=True)
    out = strata_counts(df)
    assert out.n_origens.sum() == len(df)
    assert set(out.estrato) == {*STRATA, "sem_serie"}
    assert len(out) == 6 * 3
