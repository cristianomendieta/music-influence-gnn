"""Tests for Phase-3 report figures (Fig.3 boxplot + Figs.8/9 case curves) — T9."""
from __future__ import annotations

import pandas as pd
import pytest


def _make_mode1_df() -> pd.DataFrame:
    rows = []
    songs = ["songA", "songB", "songC"]
    for chart in ["viral50", "top200"]:
        for model in ["gnn", "sir"]:
            for i, song_id in enumerate(songs):
                rows.append({
                    "song_id": song_id,
                    "chart": chart,
                    "model": model,
                    "rmse": 0.05 + 0.01 * i + (0.0 if model == "gnn" else 0.005),
                    "rmse_onchart": 0.04 + 0.01 * i,
                    "days_on_chart": 30 + 10 * i,
                    "is_long_hit": i == 0,
                    "n_weeks_eval": 12,
                    "saturation_rate": 0.1 * i,
                })
    return pd.DataFrame(rows)


def _make_curves_df(song_id: str = "songA", chart: str = "viral50") -> pd.DataFrame:
    weeks = list(range(10))
    rows = []
    for w in weeks:
        rows.append({
            "song_id": song_id,
            "chart": chart,
            "week": w,
            "y_true": 0.1 + 0.01 * w,
            "y_pred_gnn": 0.1 + 0.009 * w if w >= 4 else None,
            "y_pred_sir": 0.1 + 0.011 * w,
        })
    return pd.DataFrame(rows)


def test_fig3_boxplot_gnn_vs_sir_smoke(tmp_path):
    from music_diffusion_gnn.evaluation.figures import fig3_boxplot_gnn_vs_sir

    mode1_df = _make_mode1_df()
    out_path = tmp_path / "fig3_boxplot.png"

    result = fig3_boxplot_gnn_vs_sir(mode1_df, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_figs_8_9_cases_all_present(tmp_path):
    from music_diffusion_gnn.evaluation.figures import figs_8_9_cases

    curves_df = pd.concat([
        _make_curves_df("songA", "viral50"),
        _make_curves_df("songB", "top200"),
    ], ignore_index=True)

    cases = [
        {"song_id": "songA", "chart": "viral50", "name": "Shallow", "note": None},
        {"song_id": "songB", "chart": "top200", "name": "abcdefu", "note": None},
    ]
    out_path = tmp_path / "figs_8_9.png"

    result = figs_8_9_cases(curves_df, cases, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_figs_8_9_cases_missing_case_no_raise(tmp_path):
    """A case whose (song_id, chart) has no rows at all must not raise."""
    from music_diffusion_gnn.evaluation.figures import figs_8_9_cases

    curves_df = _make_curves_df("songA", "viral50")

    cases = [
        {"song_id": "songA", "chart": "viral50", "name": "Shallow", "note": None},
        {"song_id": "does_not_exist", "chart": "top200", "name": "Batom de Cereja", "note": None},
    ]
    out_path = tmp_path / "figs_8_9_missing.png"

    result = figs_8_9_cases(curves_df, cases, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_figs_8_9_cases_with_note_does_not_raise(tmp_path):
    """A case with a `note` (documenting a substitution) must render fine."""
    from music_diffusion_gnn.evaluation.figures import figs_8_9_cases

    curves_df = _make_curves_df("songC", "viral50")

    cases = [
        {
            "song_id": "songC",
            "chart": "viral50",
            "name": "Água Nos Zói",
            "note": "substituting for absent 'Batom de Cereja' (same duration/regime)",
        },
    ]
    out_path = tmp_path / "figs_8_9_note.png"

    result = figs_8_9_cases(curves_df, cases, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_figs_8_9_cases_creates_parent_dirs(tmp_path):
    from music_diffusion_gnn.evaluation.figures import figs_8_9_cases

    curves_df = _make_curves_df("songA", "viral50")
    cases = [{"song_id": "songA", "chart": "viral50", "name": "Shallow", "note": None}]
    out_path = tmp_path / "nested" / "dir" / "figs_8_9.png"

    figs_8_9_cases(curves_df, cases, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
