"""Graph statistics — degree distributions, components, clustering, Louvain communities."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# T11 — compute_stats
# ---------------------------------------------------------------------------

def compute_stats(hetero: HeteroData) -> dict:
    """Compute graph statistics from HeteroData.

    Returns dict with keys:
        counts: {node_type: n_nodes, edge_type_str: n_edges}
        degrees: {edge_type_str: {src: {mean,median,p95,max}, dst: ...}}
        components: {edge_type_str: n_components}
        clustering: {edge_type_str: avg_clustering}
        louvain_genre: list of (size, [genre_names]) sorted descending, top-10
    """
    stats: dict = {
        "counts": {},
        "degrees": {},
        "components": {},
        "clustering": {},
        "louvain_genre": [],
    }

    # Node counts
    for ntype in hetero.node_types:
        stats["counts"][ntype] = hetero[ntype].num_nodes

    # Edge type stats
    et_labels = {
        ("artist", "performs", "music"): "artist→performs→music",
        ("artist", "has_genre", "genre"): "artist→has_genre→genre",
        ("genre", "rev_has_genre", "artist"): "genre→rev_has_genre→artist",
        ("music", "cotrajectory", "music"): "music→cotrajectory→music",
        ("genre", "cooccurs", "genre"): "genre→cooccurs→genre",
    }

    for et in hetero.edge_types:
        ei = hetero[et].edge_index
        n_edges = ei.shape[1]
        label = et_labels.get(tuple(et), "→".join(et))
        stats["counts"][label] = n_edges

        if n_edges == 0:
            continue

        src_type, _, dst_type = et
        n_src = hetero[src_type].num_nodes
        n_dst = hetero[dst_type].num_nodes

        # Degree stats
        src_deg = torch.zeros(n_src, dtype=torch.long)
        dst_deg = torch.zeros(n_dst, dtype=torch.long)
        src_deg.scatter_add_(0, ei[0], torch.ones(n_edges, dtype=torch.long))
        dst_deg.scatter_add_(0, ei[1], torch.ones(n_edges, dtype=torch.long))

        stats["degrees"][label] = {
            "src_out": _deg_summary(src_deg.numpy()),
            "dst_in": _deg_summary(dst_deg.numpy()),
        }

        # Connected components (via NetworkX on undirected projection)
        G = nx.Graph()
        G.add_nodes_from(range(n_src))
        edges_np = ei.T.numpy().tolist()
        # Offset dst nodes to avoid collision with src in bipartite graphs
        if src_type != dst_type:
            edges_np = [(s, d + n_src) for s, d in edges_np]
        G.add_edges_from(edges_np)
        stats["components"][label] = nx.number_connected_components(G)

        # Clustering coefficient (undirected homogeneous only)
        if src_type == dst_type:
            G_hom = nx.Graph()
            G_hom.add_nodes_from(range(n_src))
            G_hom.add_edges_from(ei.T.numpy().tolist())
            stats["clustering"][label] = nx.average_clustering(G_hom)

    # Louvain on genre co-occurrence subgraph
    cooc_et = ("genre", "cooccurs", "genre")
    if cooc_et in [tuple(e) for e in hetero.edge_types]:
        ei = hetero[cooc_et].edge_index
        n_genre = hetero["genre"].num_nodes
        genre_names = hetero["genre"].genre_name if hasattr(hetero["genre"], "genre_name") else []

        G_genre = nx.Graph()
        G_genre.add_nodes_from(range(n_genre))
        if ei.shape[1] > 0:
            G_genre.add_edges_from(ei.T.numpy().tolist())

        communities = nx.algorithms.community.louvain_communities(
            G_genre, seed=42
        )
        # Sort by size descending, top-10
        communities_sorted = sorted(communities, key=len, reverse=True)[:10]
        louvain = []
        for comm in communities_sorted:
            names = [genre_names[i] for i in sorted(comm) if i < len(genre_names)]
            louvain.append((len(comm), names))
        stats["louvain_genre"] = louvain

    return stats


def _deg_summary(arr: np.ndarray) -> dict:
    arr = arr[arr > 0]  # only nodes with at least 1 edge
    if len(arr) == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


# ---------------------------------------------------------------------------
# T12 — render_report
# ---------------------------------------------------------------------------

def render_report(stats: dict, out_md: Path) -> None:
    """Render stats dict to a Markdown report at out_md."""
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Phase 1 — Graph Statistics\n"]

    # Node and edge counts
    lines.append("## Node Counts\n")
    lines.append("| Type | Count | Target | Within tolerance? |")
    lines.append("|------|-------|--------|-------------------|")
    targets = {"music": (6469, 10), "artist": (1701, 5), "genre": (530, 10)}
    for ntype, n in stats["counts"].items():
        if ntype in targets:
            target, tol = targets[ntype]
            ok = "✅" if abs(n - target) <= tol else "❌"
            lines.append(f"| {ntype} | {n} | {target}±{tol} | {ok} |")
    lines.append("")

    lines.append("## Edge Counts\n")
    lines.append("| Edge type | Count |")
    lines.append("|-----------|-------|")
    for k, v in stats["counts"].items():
        if k not in ("music", "artist", "genre"):
            lines.append(f"| {k} | {v} |")
    lines.append("")

    # Degree distributions
    lines.append("## Degree Distributions\n")
    for et_label, deg_info in stats.get("degrees", {}).items():
        lines.append(f"### {et_label}\n")
        lines.append("| Direction | Mean | Median | P95 | Max |")
        lines.append("|-----------|------|--------|-----|-----|")
        for direction, d in deg_info.items():
            lines.append(
                f"| {direction} | {d['mean']:.2f} | {d['median']:.2f} "
                f"| {d['p95']:.2f} | {d['max']} |"
            )
        lines.append("")

    # Connected components
    lines.append("## Connected Components\n")
    lines.append("| Edge type | N components |")
    lines.append("|-----------|-------------|")
    for et_label, n_comp in stats.get("components", {}).items():
        lines.append(f"| {et_label} | {n_comp} |")
    lines.append("")

    # Clustering
    lines.append("## Average Clustering Coefficient\n")
    lines.append("| Edge type | Avg clustering |")
    lines.append("|-----------|---------------|")
    for et_label, cc in stats.get("clustering", {}).items():
        lines.append(f"| {et_label} | {cc:.4f} |")
    lines.append("")

    # Louvain communities
    lines.append("## Top-10 Genre Communities (Louvain)\n")
    lines.append("| Rank | Size | Genres (sample) |")
    lines.append("|------|------|----------------|")
    for rank, (size, names) in enumerate(stats.get("louvain_genre", []), 1):
        sample = ", ".join(names[:8]) + ("..." if len(names) > 8 else "")
        lines.append(f"| {rank} | {size} | {sample} |")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written: %s", out_md)


# ---------------------------------------------------------------------------
# T13 — plot_degree_distributions
# ---------------------------------------------------------------------------

def plot_degree_distributions(hetero: HeteroData, out_png: Path) -> None:
    """Plot 4-panel log-log degree distribution figure (one panel per edge type)."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    edge_types_plot = [
        (("artist", "performs", "music"), "artist→performs→music", "dst"),
        (("artist", "has_genre", "genre"), "artist→has_genre→genre", "src"),
        (("music", "cotrajectory", "music"), "music→cotrajectory→music", "src"),
        (("genre", "cooccurs", "genre"), "genre→cooccurs→genre", "src"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for ax, (et, title, degree_dir) in zip(axes_flat, edge_types_plot):
        et_tuple = tuple(et)
        if et_tuple not in [tuple(e) for e in hetero.edge_types]:
            ax.set_visible(False)
            continue

        ei = hetero[et_tuple].edge_index
        n_edges = ei.shape[1]

        src_type, _, dst_type = et
        n_src = hetero[src_type].num_nodes
        n_dst = hetero[dst_type].num_nodes

        if n_edges == 0:
            ax.text(0.5, 0.5, "No edges", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        if degree_dir == "src":
            deg = torch.zeros(n_src, dtype=torch.long)
            deg.scatter_add_(0, ei[0], torch.ones(n_edges, dtype=torch.long))
            label = "out-degree (src)"
        else:
            deg = torch.zeros(n_dst, dtype=torch.long)
            deg.scatter_add_(0, ei[1], torch.ones(n_edges, dtype=torch.long))
            label = "in-degree (dst)"

        deg_np = deg.numpy()
        deg_np = deg_np[deg_np > 0]

        if len(deg_np) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Log-log histogram
        bins = np.logspace(np.log10(max(1, deg_np.min())), np.log10(deg_np.max() + 1), 30)
        counts, bin_edges = np.histogram(deg_np, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mask = counts > 0
        ax.loglog(bin_centers[mask], counts[mask], "o-", markersize=4, lw=1.5)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Phase 1 — Degree Distributions (log-log)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", out_png)
