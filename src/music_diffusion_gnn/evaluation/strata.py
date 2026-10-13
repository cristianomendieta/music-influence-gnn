"""Life-stage strata of forecast origins, read off the on-chart weeks (Phase 6b).

Every stratum is a function of the on-chart weeks ``<= origin_week`` only, so the
map built on top of it is usable at prediction time and never conditions on the
outcome. Defined in ``.specs/features/phase-6b-mapa-estrutura/spec.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATA = ("estreia", "reentrada", "jovem", "maduro_mono", "maduro_multi")
SEM_SERIE = "sem_serie"   # origin whose (song, chart) has no row in timeseries.parquet


def assign_strata(
    origins: pd.DataFrame,
    onchart: set[tuple[str, str, int]],
    *,
    young_weeks: int = 8,
    wave_gap: int = 4,
    series_keys: set[tuple[str, str]] | None = None,
) -> pd.Series:
    """Stratum of each ``(song_id, chart, origin_week)`` row of ``origins``.

    ``onchart`` is the ``(song_id, chart, week)`` set from
    :func:`evaluation.longhits.onchart_weeks`.

    - ``estreia``: no on-chart week ``<= w``
    - ``reentrada``: ``w`` off-chart, with an earlier on-chart week
    - ``jovem``: ``w`` on-chart, ``w - first on-chart week <= young_weeks``
    - ``maduro_mono`` / ``maduro_multi``: ``w`` on-chart, older than that, with one
      or several waves up to ``w``. Two on-chart weeks belong to different waves
      when at least ``wave_gap`` consecutive off-chart weeks separate them.

    When ``series_keys`` (the ``(song_id, chart)`` pairs present in the time
    series) is given, an origin outside it is ``sem_serie`` instead of ``estreia``:
    "never seen on chart" and "not in the data at all" are different things.
    """
    stages = np.empty(len(origins), dtype=object)
    stages[:] = "estreia"

    weeks_by_series: dict[tuple[str, str], np.ndarray] = {}
    if onchart:
        oc = pd.DataFrame(list(onchart), columns=["song_id", "chart", "week"])
        weeks_by_series = {
            key: np.sort(grp["week"].to_numpy(dtype=np.int64))
            for key, grp in oc.groupby(["song_id", "chart"], sort=False)
        }

    positions = pd.RangeIndex(len(origins))
    for key, pos in pd.Series(positions).groupby(
        [origins["song_id"].to_numpy(), origins["chart"].to_numpy()], sort=False
    ):
        on = weeks_by_series.get(key)
        if on is None:
            continue
        idx = pos.to_numpy()
        w = origins["origin_week"].to_numpy()[idx].astype(np.int64)

        n_prior = np.searchsorted(on, w, side="right")
        has_prior = n_prior > 0
        is_on = np.isin(w, on)

        # wave id of every on-chart week: a new wave opens after >= wave_gap off weeks
        new_wave = np.concatenate([[0], (np.diff(on) - 1) >= wave_gap]).astype(np.int64)
        wave_of = np.cumsum(new_wave)
        n_waves = np.where(has_prior, wave_of[np.maximum(n_prior - 1, 0)] + 1, 0)
        age = w - on[0]

        stage = np.select(
            [
                ~has_prior,
                ~is_on,
                age <= young_weeks,
                n_waves == 1,
            ],
            ["estreia", "reentrada", "jovem", "maduro_mono"],
            default="maduro_multi",
        )
        stages[idx] = stage

    if series_keys is not None:
        known = pd.MultiIndex.from_arrays(
            [origins["song_id"].to_numpy(), origins["chart"].to_numpy()]
        ).isin(list(series_keys))
        stages[~known] = SEM_SERIE

    return pd.Series(stages, index=origins.index, name="estrato")


def strata_counts(origins: pd.DataFrame, *, min_songs: int = 30) -> pd.DataFrame:
    """Origins and distinct songs per ``(split_regime, estrato, k)`` cell.

    ``origins`` needs ``split_regime``, ``estrato``, ``k`` and ``song_id`` (one row
    per origin, a single model's rows). A song in both charts counts once in
    ``n_musicas``. ``insuficiente`` is true when ``n_musicas < min_songs``: the
    cell is reported but kept out of the tests and the Holm family.
    """
    out = (
        origins.groupby(["split_regime", "estrato", "k"], observed=True)
        .agg(n_origens=("song_id", "size"), n_musicas=("song_id", "nunique"))
        .reset_index()
    )
    out["insuficiente"] = out["n_musicas"] < min_songs
    return out
