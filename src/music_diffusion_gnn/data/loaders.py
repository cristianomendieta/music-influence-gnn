"""Raw data loaders — CSV → DataFrame, normalized types."""
from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"


def load_charts(path: Path | None = None) -> pd.DataFrame:
    """Load Spotify BR charts (Top 200 + Viral 50) from data/charts/mgdplus/.

    Always returns a DataFrame with canonical columns:
        song_id | date | rank | chart | streams
    where chart ∈ {top200, viral50}.
    """
    mgdplus_dir = path if path is not None else DATA / "charts" / "mgdplus"
    if not mgdplus_dir.exists() or not any(mgdplus_dir.glob("*.csv")):
        raise FileNotFoundError(
            f"MGD+ charts directory not found or empty: {mgdplus_dir}\n"
            "Expected: data/charts/mgdplus/spotify_charts_regional_br.csv + spotify_charts_viral_br.csv"
        )
    return _load_charts_mgdplus(mgdplus_dir)


def _load_charts_mgdplus(charts_dir: Path) -> pd.DataFrame:
    """Load MGD+ daily chart CSVs from data/charts/mgdplus/."""
    files = sorted(charts_dir.glob("*.csv"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, sep="\t", low_memory=False)
        except Exception:
            df = pd.read_csv(f, low_memory=False)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    # Normalise column names (MGD+ uses: ID, date, Position, Chart, Streams, Trend, Track, Artist)
    raw.columns = [c.strip() for c in raw.columns]
    col_lower = {c.lower(): c for c in raw.columns}

    def _get(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in col_lower:
                return col_lower[c]
        return None

    id_col = _get(["id", "song_id", "track_id"])
    date_col = _get(["date"])
    rank_col = _get(["position", "rank"])
    chart_col = _get(["chart"])
    stream_col = _get(["streams", "stream"])

    out = pd.DataFrame()
    # song_id: extract from URL if needed
    if id_col:
        sample = str(raw[id_col].dropna().iloc[0])
        if "spotify.com/track/" in sample:
            out["song_id"] = raw[id_col].str.extract(r"/track/([A-Za-z0-9]+)")[0]
        else:
            out["song_id"] = raw[id_col]

    out["date"] = pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.NaT
    out["rank"] = pd.to_numeric(raw[rank_col], errors="coerce") if rank_col else None
    out["streams"] = pd.to_numeric(raw[stream_col], errors="coerce") if stream_col else None

    # Normalise chart type labels to {top200, viral50}
    if chart_col:
        chart_raw = raw[chart_col].str.strip().str.lower()
        out["chart"] = (
            chart_raw
            .str.replace(r"top.?200", "top200", regex=True)
            .str.replace(r"viral.?50", "viral50", regex=True)
        )
    else:
        out["chart"] = "top200"

    return out.dropna(subset=["song_id", "date"]).reset_index(drop=True)



def load_songs(path: Path | None = None) -> pd.DataFrame:
    """Load MGD+ acoustic features for hit songs (all years combined).

    Returns deduplicated DataFrame keyed by song_id.
    """
    if path is None:
        path = DATA / "songs"
    if isinstance(path, Path) and path.is_dir():
        files = sorted(path.glob("br-hit_songs-*.csv"))
        df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    else:
        df = pd.read_csv(path, sep="\t")
    return df.drop_duplicates(subset=["song_id"]).reset_index(drop=True)


def load_release_dates(path: Path | None = None) -> pd.Series:
    """Map song_id → release_date (datetime). Missing/unparseable → NaT.

    Used to bound each song's time series at its actual release per
    Oliveira et al. 2025 §3.2.
    """
    df = load_songs(path)
    rd = pd.to_datetime(df["release_date"], errors="coerce", format="mixed")
    return pd.Series(rd.values, index=df["song_id"].values, name="release_date")


def load_artists(path: Path | None = None) -> pd.DataFrame:
    """Load artist metadata with genre lists parsed to Python lists."""
    if path is None:
        path = DATA / "artists" / "br-artists-all_time.csv"
    df = pd.read_csv(path, sep="\t")
    df["genres_list"] = df["genres"].apply(ast.literal_eval)
    return df


def load_genre_network(
    years: int | Iterable[int] | None = None, path: Path | None = None
) -> pd.DataFrame:
    """Load genre↔genre co-occurrence edges: ``Source | Target | Weight``.

    These CSVs are **comma**-separated (unlike the rest of MGD+, which is
    tab-separated). ``years`` selects which yearly files to concatenate
    (all of them when ``None``).

    ``Avg_Popularity`` and ``Avg_Streams`` are never read: they aggregate the
    whole 2017-2021 window and leak the target (ADR-0003).
    """
    cols = ["Source", "Target", "Weight"]
    if path is not None:
        return pd.read_csv(path, usecols=cols)
    base = DATA / "genre_network"
    if years is None:
        files = sorted(base.glob("br-genre_network-*.csv"))
    else:
        if isinstance(years, int):
            years = [years]
        files = [base / f"br-genre_network-{y}.csv" for y in years]
    return pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)


def load_artists_years(years: Iterable[int]) -> pd.DataFrame:
    """Load the per-year artist files for ``years`` (artist_id, genres).

    One row per artist that charted in that year; an artist appearing in more
    than one year is kept once (their genre tags do not change between files).
    Used by the genre attributes, which must see only the training years.
    """
    df = pd.concat(
        [
            pd.read_csv(
                DATA / "artists" / f"br-artists-{y}.csv",
                sep="\t",
                usecols=["artist_id", "genres"],
            )
            for y in years
        ],
        ignore_index=True,
    ).drop_duplicates("artist_id")
    df["genres_list"] = df["genres"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else []
    )
    return df
