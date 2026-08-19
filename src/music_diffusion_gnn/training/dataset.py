"""Dataset utilities: weekly aggregation, temporal splits, and causal window sampling."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from music_diffusion_gnn.graph.temporal import week_index

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Split regimes (ADR 0005): named, selectable temporal boundaries
# ---------------------------------------------------------------------------
# Note: the formula (iso_year - 2017)*52 + (iso_week - 1) is not bijective
# for years with 53 ISO weeks (e.g. 2020). The dates 2020-06-30 and
# 2020-07-01 both map to week 182; 2020-12-31 and 2021-01-01 both map to
# 208. Each regime therefore pins explicit boundary dates that land cleanly
# on one side of any such ambiguity.


@dataclass(frozen=True)
class SplitRegime:
    """A named train/val/test week-boundary configuration.

    ``val`` is the open interval strictly between ``train_end_week`` and
    ``test_start_week``. ``test_end_week=None`` means the test split is
    open-ended (runs to the end of available data); a finite value bounds it
    (used by the pre-pandemia regime, which must not bleed into 2020+ data).
    """

    name: str
    train_end_week: int
    val_start_week: int
    val_end_week: int
    test_start_week: int
    test_end_week: int | None


SPLIT_REGIMES: dict[str, SplitRegime] = {
    # Qualificação split: train ≤ 2020-06-30, val 2020-07..2020-12, test 2021+ (open-ended).
    "current": SplitRegime(
        name="current",
        train_end_week=week_index("2020-06-30"),   # 182 (ISO 2020-W27)
        val_start_week=week_index("2020-06-30") + 1,
        val_end_week=week_index("2020-12-31") - 1,
        test_start_week=week_index("2020-12-31"),   # 208 (ISO 2020-W53/2021-W01)
        test_end_week=None,
    ),
    # Robustness check (ADR 0005): entirely pre-pandemic, train 2017-2018,
    # val H1 2019, test H2 2019 — bounded so 2020+ data never enters any split.
    "pre_pandemia": SplitRegime(
        name="pre_pandemia",
        train_end_week=week_index("2018-12-30"),    # 2018-W52
        val_start_week=week_index("2018-12-31"),     # 2019-W01
        val_end_week=week_index("2019-06-30"),        # 2019-W26
        test_start_week=week_index("2019-07-01"),     # 2019-W27
        test_end_week=week_index("2019-12-29"),        # 2019-W52
    ),
}
DEFAULT_SPLIT_REGIME = "current"


def get_split_regime(regime: str | SplitRegime = DEFAULT_SPLIT_REGIME) -> SplitRegime:
    """Resolve a regime name (or pass-through a ``SplitRegime``) via ``SPLIT_REGIMES``."""
    if isinstance(regime, SplitRegime):
        return regime
    try:
        return SPLIT_REGIMES[regime]
    except KeyError:
        raise ValueError(
            f"Unknown split regime {regime!r}; available: {sorted(SPLIT_REGIMES)}"
        ) from None


# Backward-compatible module-level aliases — equal to the "current" regime,
# kept because they're imported directly across evaluation/training code.
TRAIN_END_WEEK  = SPLIT_REGIMES["current"].train_end_week   # 182
TEST_START_WEEK = SPLIT_REGIMES["current"].test_start_week  # 208
# train : week <= 182
# val   : 183 <= week <= 207
# test  : week >= 208


# ---------------------------------------------------------------------------
# T2: aggregate_weekly
# ---------------------------------------------------------------------------

def aggregate_weekly(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily timeseries to weekly targets.

    Reads the daily ``(song_id, chart, date, y)`` DataFrame and returns
    ``(song_id, chart, week, y_week)`` where ``y_week`` is the mean of
    daily ``y`` values within each ISO-week. Rows where ``week > 260``
    (dates in 2022) are discarded.
    """
    df = ts_df.copy()

    # Vectorized week_index: (iso_year - 2017) * 52 + (iso_week - 1)
    # Uses pandas isocalendar() to avoid row-by-row ValueError on ISO-year boundaries
    # (e.g. 2017-01-01 is ISO 2016-W52, which maps to -1 and must be discarded).
    iso = df["date"].dt.isocalendar()
    df["week"] = (iso["year"].astype(int) - 2017) * 52 + (iso["week"].astype(int) - 1)

    # Discard weeks outside the graph range [0, 260]
    df = df[(df["week"] >= 0) & (df["week"] <= 260)]

    # Aggregate: mean of y per (song_id, chart, week)
    weekly = (
        df.groupby(["song_id", "chart", "week"], observed=True)["y"]
        .mean()
        .reset_index()
        .rename(columns={"y": "y_week"})
    )

    assert {"song_id", "chart", "week", "y_week"} <= set(weekly.columns)
    assert weekly["week"].max() <= 260
    assert weekly["y_week"].between(0, 0.5).all(), (
        f"y_week out of [0,0.5]: min={weekly['y_week'].min()}, max={weekly['y_week'].max()}"
    )
    return weekly


# ---------------------------------------------------------------------------
# T3: temporal_split
# ---------------------------------------------------------------------------

def temporal_split(
    weekly_df: pd.DataFrame, regime: str | SplitRegime = DEFAULT_SPLIT_REGIME
) -> dict[str, pd.DataFrame]:
    """Split weekly DataFrame into train / val / test by week boundary.

    ``regime`` selects a named :class:`SplitRegime` from ``SPLIT_REGIMES``
    (default ``"current"``, identical to the boundaries used at qualificação).
    Passing ``"current"`` (or omitting ``regime``) reproduces today's numbers
    exactly; other regimes (e.g. ``"pre_pandemia"``) use their own bounded
    train/val/test windows.

    Splits are pairwise disjoint. For the default open-ended regime their
    union equals the full DataFrame; a bounded regime instead drops rows
    outside its window (they belong to no split).
    """
    r = get_split_regime(regime)

    train = weekly_df[weekly_df["week"] <= r.train_end_week].copy()
    val   = weekly_df[
        (weekly_df["week"] >= r.val_start_week) & (weekly_df["week"] <= r.val_end_week)
    ].copy()
    if r.test_end_week is None:
        test = weekly_df[weekly_df["week"] >= r.test_start_week].copy()
    else:
        test = weekly_df[
            (weekly_df["week"] >= r.test_start_week) & (weekly_df["week"] <= r.test_end_week)
        ].copy()

    # Verify clean partition (no week appears in more than one split)
    assert set(train["week"].unique()).isdisjoint(set(val["week"].unique()))
    assert set(val["week"].unique()).isdisjoint(set(test["week"].unique()))
    assert set(train["week"].unique()).isdisjoint(set(test["week"].unique()))
    if r.test_end_week is None:
        # Open-ended regime: union covers all rows (unchanged current behavior)
        assert len(train) + len(val) + len(test) == len(weekly_df)

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# T4: Sample dataclass + build_samples
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    song_idx: int           # PyG node index for the music node
    chart: int              # 0=viral50, 1=top200
    target_week: int        # w ∈ [1, 260]
    window_weeks: list[int] # [w-W, ..., w-1], left-padded with -1
    pad_mask: list[bool]    # True where entry is padding (week < first_seen_week or w-k < 0)
    y: float                # y_week observed ∈ [0, 0.5]


_CHART_CODE = {"viral50": 0, "top200": 1}


def build_samples(
    weekly_df: pd.DataFrame,
    W: int,
    node_id_map_path: Path | str,
    first_seen: dict[tuple[str, str], int] | None = None,
) -> list[Sample]:
    """Build causal windowed training samples from weekly aggregated DataFrame.

    Args:
        weekly_df: output of ``aggregate_weekly`` (song_id, chart, week, y_week)
        W: look-back window length in weeks
        node_id_map_path: path to ``node_id_map.json`` (for song_id → PyG index)
        first_seen: optional pre-computed dict ``{(song_id, chart): first_week}``;
            computed from ``weekly_df`` if None.

    Returns:
        List of ``Sample`` objects; one per (song_id, chart, target_week) tuple
        where ``target_week > first_seen_week`` for that (song, chart).
    """
    # Load node_id_map
    with open(node_id_map_path) as f:
        nmap = json.load(f)
    song_to_idx: dict[str, int] = nmap["music"]["spotify_id_to_idx"]

    # Compute first_seen_week per (song_id, chart) if not provided
    if first_seen is None:
        fs = (
            weekly_df.groupby(["song_id", "chart"], observed=True)["week"]
            .min()
            .to_dict()
        )
        first_seen = {(sid, chart): w for (sid, chart), w in fs.items()}

    # Index for fast lookup: (song_id, chart, week) → y_week
    weekly_indexed = weekly_df.set_index(["song_id", "chart", "week"])["y_week"]

    # Validate all songs are in graph
    missing = set(weekly_df["song_id"].unique()) - set(song_to_idx.keys())
    assert not missing, (
        f"{len(missing)} song_ids not in node_id_map — "
        "C4 of Phase 1 should guarantee all subset songs are in the graph: "
        f"{list(missing)[:5]}"
    )

    # Vectorized: add song_idx, chart_code, first_seen_week columns
    df = weekly_df.copy()
    df["song_idx"]   = df["song_id"].map(song_to_idx)
    df["chart_code"] = df["chart"].map(_CHART_CODE)
    df["fsw"]        = df.apply(lambda r: first_seen[(r["song_id"], r["chart"])], axis=1)

    # Filter: target only for week > first_seen_week
    df = df[df["week"] > df["fsw"]].copy()

    # Build window columns (W offsets) as numpy arrays — avoids per-row Python loops
    weeks_arr = df["week"].to_numpy()
    fsw_arr   = df["fsw"].to_numpy()

    # window_weeks[k] = w - (W - k), for k in 0..W-1
    #   i.e. offsets: w-W, w-W+1, ..., w-1
    offsets = np.arange(W, 0, -1)  # [W, W-1, ..., 1]
    # Shape: (N, W)
    wk_matrix = weeks_arr[:, None] - offsets[None, :]  # (N, W)

    # Padding: where wk < fsw or wk < 0
    fsw_matrix = fsw_arr[:, None]
    pad_matrix = (wk_matrix < fsw_matrix) | (wk_matrix < 0)
    # Replace padded positions with -1
    wk_matrix[pad_matrix] = -1

    # Build list of Samples
    song_idx_arr  = df["song_idx"].to_numpy(dtype=np.int64)
    chart_code_arr = df["chart_code"].to_numpy(dtype=np.int64)
    target_week_arr = weeks_arr
    y_arr = df["y_week"].to_numpy(dtype=np.float64)

    samples: list[Sample] = []
    for i in range(len(df)):
        samples.append(Sample(
            song_idx=int(song_idx_arr[i]),
            chart=int(chart_code_arr[i]),
            target_week=int(target_week_arr[i]),
            window_weeks=wk_matrix[i].tolist(),
            pad_mask=pad_matrix[i].tolist(),
            y=float(y_arr[i]),
        ))

    return samples


# ---------------------------------------------------------------------------
# R1.T1: build_pop_bank — dense per-week popularity for node-feature injection
# ---------------------------------------------------------------------------

def build_pop_bank(
    weekly_df: pd.DataFrame,
    node_id_map_path: Path | str,
    n_music: int,
    n_weeks: int = 261,
) -> torch.Tensor:
    """Build a dense per-week popularity tensor for node-feature injection (R1).

    Returns a tensor ``pop_bank`` of shape ``(n_weeks, n_music, 2)`` where
    ``pop_bank[w, song_idx, chart_code] = y_week(song, chart, w)`` and ``0.0``
    where the (song, chart) is absent from the chart in week ``w``.

    Channel order matches ``_CHART_CODE``: column 0 = viral50, column 1 = top200.
    Row order matches the PyG music node index from ``node_id_map.json`` so it
    aligns with the encoder's ``Z_music`` and with ``bank[w][song_idxs]``.

    Built from the *full* ``weekly_df`` (all splits), so ``pop_bank[w-1, …]``
    equals the naive persistence value (with the same 0.0 floor for gap weeks)
    used by :func:`persistence_predict` — the residual head can therefore be
    anchored to persistence exactly.

    Args:
        weekly_df: output of :func:`aggregate_weekly` (song_id, chart, week, y_week).
        node_id_map_path: path to ``node_id_map.json`` (song_id → PyG index).
        n_music: number of music nodes in the graph (``g["music"].num_nodes``).
        n_weeks: number of week slots; default 261 covers weeks ``[0, 260]``.

    Returns:
        ``torch.float32`` tensor of shape ``(n_weeks, n_music, 2)``.
    """
    with open(node_id_map_path) as f:
        nmap = json.load(f)
    song_to_idx: dict[str, int] = nmap["music"]["spotify_id_to_idx"]

    pop = torch.zeros((n_weeks, n_music, 2), dtype=torch.float32)

    # Vectorised scatter via numpy index arrays
    df = weekly_df[weekly_df["week"] < n_weeks].copy()
    song_idx = df["song_id"].map(song_to_idx)
    valid = song_idx.notna()
    if not valid.all():
        df = df[valid.values]
        song_idx = song_idx[valid.values]

    w_arr   = df["week"].to_numpy(dtype=np.int64).copy()
    s_arr   = song_idx.to_numpy(dtype=np.int64).copy()
    c_arr   = df["chart"].map(_CHART_CODE).to_numpy(dtype=np.int64).copy()
    y_arr   = df["y_week"].to_numpy(dtype=np.float32).copy()

    pop[
        torch.from_numpy(w_arr),
        torch.from_numpy(s_arr),
        torch.from_numpy(c_arr),
    ] = torch.from_numpy(y_arr)

    return pop
