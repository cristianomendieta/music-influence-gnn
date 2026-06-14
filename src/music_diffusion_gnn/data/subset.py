"""Build and persist the viral∩hit subset."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from music_diffusion_gnn.data.loaders import load_charts, load_songs

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"


def build_subset(
    out_path: Path | None = None,
    *,
    charts_df: pd.DataFrame | None = None,
    songs_df: pd.DataFrame | None = None,
    require_acoustic: bool = False,
) -> list[str]:
    """Return sorted list of song_ids in viral∩hit (optionally filtered to songs with acoustic features).

    Persists data/processed/subset_ids.json and returns the list.
    require_acoustic=True matches the old behaviour; False matches the paper (Oliveira et al. 2025).
    """
    if out_path is None:
        out_path = PROCESSED / "subset_ids.json"

    if charts_df is None:
        charts_df = load_charts()

    top_ids: set[str] = set(charts_df.loc[charts_df["chart"] == "top200", "song_id"].dropna())
    viral_ids: set[str] = set(charts_df.loc[charts_df["chart"] == "viral50", "song_id"].dropna())

    if require_acoustic:
        if songs_df is None:
            songs_df = load_songs()
        acoustic_ids: set[str] = set(songs_df["song_id"].dropna())
        subset = sorted(top_ids & viral_ids & acoustic_ids)
    else:
        subset = sorted(top_ids & viral_ids)

    assert len(subset) > 0, "Subset is empty — check data paths"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "viral_intersect_hit": subset,
        "n": len(subset),
        "require_acoustic": require_acoustic,
        "generated_at": str(date.today()),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return subset


def load_subset(path: Path | None = None) -> list[str]:
    """Load subset from JSON; raise FileNotFoundError if not yet generated."""
    if path is None:
        path = PROCESSED / "subset_ids.json"
    data = json.loads(path.read_text())
    return data["viral_intersect_hit"]
