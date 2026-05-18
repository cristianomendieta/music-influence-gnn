"""Phase 1 orchestrator — build HeteroGraph, stats, smoke-test, C1-C9 checklist.

Usage:
    python scripts/run_phase1.py [--force]

--force : rebuild even if hetero_full.pt already exists
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_GRAPH = ROOT / "data" / "processed" / "graph"
RESULTS = ROOT / "results" / "phase1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_phase1")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f}min" if s > 60 else f"{s:.1f}s"


def _banner(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def _check(criterion: str, ok: bool, detail: str = "") -> bool:
    icon = "\033[92m✓\033[0m" if ok else "\033[91m✘\033[0m"
    msg = f"  {icon} {criterion}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return ok


def main(force: bool = False) -> int:
    t_total = time.time()
    PROCESSED_GRAPH.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    all_pass = True

    # ------------------------------------------------------------------ #
    # Step 1 — Build graph
    # ------------------------------------------------------------------ #
    _banner("Step 1/3 — Build HeteroData")
    pt_path = PROCESSED_GRAPH / "hetero_full.pt"

    if force or not pt_path.exists():
        from music_diffusion_gnn.graph.build import build_hetero
        t0 = time.time()
        try:
            g = build_hetero(PROCESSED_GRAPH)
            print(f"  Build complete [{_elapsed(t0)}]")
        except AssertionError as exc:
            print(f"\033[91m  BUILD FAILED: {exc}\033[0m")
            return 1
    else:
        import torch
        print(f"  Cached: {pt_path}")
        g = torch.load(pt_path, weights_only=False)

    # ------------------------------------------------------------------ #
    # Step 2 — Stats
    # ------------------------------------------------------------------ #
    _banner("Step 2/3 — Graph statistics")
    t0 = time.time()
    from music_diffusion_gnn.graph.stats import (
        compute_stats,
        plot_degree_distributions,
        render_report,
    )
    stats = compute_stats(g)
    render_report(stats, RESULTS / "stats.md")
    plot_degree_distributions(g, RESULTS / "degree_distributions.png")
    print(f"  Stats complete [{_elapsed(t0)}]")

    # ------------------------------------------------------------------ #
    # Step 3 — Smoke test + C1-C9 checklist
    # ------------------------------------------------------------------ #
    _banner("Step 3/3 — Smoke test & checklist C1-C9")

    counts = stats["counts"]
    n_music = counts.get("music", 0)
    n_artist = counts.get("artist", 0)
    n_genre = counts.get("genre", 0)

    subset_path = ROOT / "data" / "processed" / "subset_ids.json"
    if subset_path.exists():
        from music_diffusion_gnn.data.subset import load_subset
        subset_ids = load_subset(subset_path)
        music_id_map = {s: i for i, s in enumerate(g["music"].song_id)}
        subset_in_graph = [s for s in subset_ids if s in music_id_map]
    else:
        subset_ids = []
        music_id_map = {}
        subset_in_graph = []

    # C1-C3 from counts
    all_pass &= _check("C1 n_music=6469±100", abs(n_music - 6469) <= 100, f"n_music={n_music}")
    all_pass &= _check("C2 n_artist=1701±5", abs(n_artist - 1701) <= 5, f"n_artist={n_artist}")
    all_pass &= _check("C3 n_genre=530±10", abs(n_genre - 530) <= 10, f"n_genre={n_genre}")

    # C4 — subset coverage
    if subset_ids:
        missing = len(subset_ids) - len(subset_in_graph)
        all_pass &= _check("C4 subset⊆music", missing == 0, f"missing={missing}/{len(subset_ids)}")
    else:
        _check("C4 subset⊆music", True, "subset_ids.json not found — skipped")

    # C5 — subset artists reachable
    if subset_ids:
        perf_et = ("artist", "performs", "music")
        if perf_et in [tuple(e) for e in g.edge_types]:
            perf_ei = g[perf_et].edge_index
            music_with_edges = set(perf_ei[1].tolist())
            subset_idxs = {music_id_map[s] for s in subset_in_graph}
            isolated = subset_idxs - music_with_edges
            all_pass &= _check(
                "C5 subset artists reachable",
                len(isolated) == 0,
                f"isolated={len(isolated)}",
            )
        else:
            _check("C5 subset artists reachable", True, "no performs edges — skipped")

    # C6 — no dangling edges
    try:
        from music_diffusion_gnn.graph.build import _validate
        _validate(g, music_id_map, {a: i for i, a in enumerate(g["artist"].artist_id)},
                  {gn: i for i, gn in enumerate(g["genre"].genre_name)})
        all_pass &= _check("C6 no dangling edges", True)
    except AssertionError as exc:
        all_pass &= _check("C6 no dangling edges", False, str(exc)[:80])

    # C7 — first_seen_week in [0, 260]
    import torch
    c7_ok = True
    c7_detail = ""
    for et in g.edge_types:
        store = g[et]
        keys = set(store.keys())
        if "edge_attr" in keys and store.edge_attr is not None and store.edge_attr.shape[1] > 0:
            fsw = store.edge_attr[:, -1]
        elif "first_seen_week" in keys:
            fsw = store.first_seen_week.float()
        else:
            continue
        if not ((fsw >= 0).all() and (fsw <= 260).all()):
            c7_ok = False
            c7_detail = f"{et}: min={fsw.min().item()}, max={fsw.max().item()}"
            break
    all_pass &= _check("C7 first_seen_week∈[0,260]", c7_ok, c7_detail)

    # C8 — HeteroSAGE smoke test (2 layers, hidden=128)
    c8_ok = False
    c8_detail = ""
    try:
        from torch_geometric.nn import HeteroConv, SAGEConv
        import torch.nn as nn

        HIDDEN = 128
        conv1 = HeteroConv(
            {et: SAGEConv((-1, -1), HIDDEN) for et in g.edge_types}, aggr="sum"
        )
        conv2 = HeteroConv(
            {et: SAGEConv((-1, -1), HIDDEN) for et in g.edge_types}, aggr="sum"
        )

        x_dict = {k: v for k, v in g.x_dict.items()}
        ei_dict = {k: v for k, v in g.edge_index_dict.items()}

        with torch.no_grad():
            x1 = conv1(x_dict, ei_dict)
            x1 = {k: v.relu() for k, v in x1.items()}
            out = conv2(x1, ei_dict)

        shape = out["music"].shape
        c8_ok = shape[1] == HIDDEN
        c8_detail = f"music embedding shape={tuple(shape)}"
    except Exception as exc:
        c8_detail = str(exc)[:120]
    all_pass &= _check("C8 HeteroSAGE forward", c8_ok, c8_detail)

    # C9 — mask_until monotonic
    c9_ok = False
    c9_detail = ""
    try:
        from music_diffusion_gnn.graph.temporal import mask_until
        cotraj_et = ("music", "cotrajectory", "music")
        g260 = mask_until(g, 260)
        g130 = mask_until(g, 130)
        e260 = g260[cotraj_et].edge_index.shape[1]
        e130 = g130[cotraj_et].edge_index.shape[1]
        c9_ok = e130 <= e260
        c9_detail = f"e(w=130)={e130} <= e(w=260)={e260}"
    except Exception as exc:
        c9_detail = str(exc)[:100]
    all_pass &= _check("C9 mask_until monotonic", c9_ok, c9_detail)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    total_elapsed = _elapsed(t_total)
    print(f"\n{'='*60}")
    if all_pass:
        print(f"\033[92m  ALL C1-C9 PASSED  [{total_elapsed}]\033[0m")
    else:
        print(f"\033[91m  SOME CRITERIA FAILED  [{total_elapsed}]\033[0m")
    print(f"{'='*60}")
    print(f"  Artifacts:")
    print(f"    {PROCESSED_GRAPH / 'hetero_full.pt'}")
    print(f"    {PROCESSED_GRAPH / 'node_id_map.json'}")
    print(f"    {RESULTS / 'stats.md'}")
    print(f"    {RESULTS / 'degree_distributions.png'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: build heterogeneous graph")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    args = parser.parse_args()
    sys.exit(main(force=args.force))
