"""MusicDiffusionGNN: encoder + temporal head orchestrator with per-week embedding bank."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from music_diffusion_gnn.graph.temporal import mask_until
from music_diffusion_gnn.models.encoder import HeteroSpatialEncoder
from music_diffusion_gnn.models.temporal_head import TemporalHead

if TYPE_CHECKING:
    from music_diffusion_gnn.training.dataset import Sample


_COTRAJ_ET = ("music", "cotrajectory", "music")


def _subsample_cotraj(snap: HeteroData, max_edges: int) -> HeteroData:
    """Randomly subsample cotrajectory edges if there are more than max_edges.

    Returns the same object if cotrajectory is already within budget.
    """
    if _COTRAJ_ET not in snap.edge_types:
        return snap
    store = snap[_COTRAJ_ET]
    n_edges = store.edge_index.shape[1]
    if n_edges <= max_edges:
        return snap

    perm = torch.randperm(n_edges)[:max_edges]
    snap[_COTRAJ_ET].edge_index = store.edge_index[:, perm]
    if "edge_attr" in store and store.edge_attr is not None:
        snap[_COTRAJ_ET].edge_attr = store.edge_attr[perm]
    if "first_seen_week" in store:
        snap[_COTRAJ_ET].first_seen_week = store.first_seen_week[perm]
    return snap


class MusicDiffusionGNN(nn.Module):
    """Heterogeneous temporal GNN for music popularity prediction.

    Architecture:
        encode_weeks: snapshot per week via mask_until → HeteroSAGE → Z_music
        predict: gather window sequences for each sample → GRU+MLP → ŷ ∈ [0,0.5]

    The embedding bank is computed *once per distinct week per forward call*,
    cached intra-forward (not between epochs — weights change), and is fully
    part of the autograd graph.
    """

    def __init__(
        self,
        metadata: tuple,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.encoder = HeteroSpatialEncoder(metadata, hidden=hidden, layers=layers, dropout=dropout)
        self.head = TemporalHead(hidden=hidden, dropout=dropout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_weeks(
        self,
        g: HeteroData,
        weeks: list[int],
        max_cotraj_edges: int | None = None,
    ) -> dict[int, Tensor]:
        """Compute music embeddings for each distinct week (one forward per week).

        Args:
            g: the full ``hetero_full`` graph
            weeks: list of week indices to encode (duplicates encoded once)
            max_cotraj_edges: if given, randomly subsample cotrajectory edges to
                at most this many per snapshot (DropEdge regularization).
                Reduces autograd memory when the full graph exceeds RAM.

        Returns:
            ``{week: Z_music}`` where each Z_music ∈ (N_music, hidden)
        """
        bank: dict[int, Tensor] = {}
        for w in set(weeks):
            snap = mask_until(g, w)
            if max_cotraj_edges is not None:
                snap = _subsample_cotraj(snap, max_cotraj_edges)
            bank[w] = self.encoder(snap.x_dict, snap.edge_index_dict)
        return bank

    def predict(
        self,
        bank: dict[int, Tensor],
        samples: list[Sample],
    ) -> Tensor:
        """Assemble window sequences and predict rank scores.

        Uses vectorised advanced indexing (one tensor op per window position)
        instead of per-sample Python loops — critical for large week-grouped batches.

        Args:
            bank: ``{week: Z_music}`` from ``encode_weeks``
            samples: list of Sample objects (same batch)

        Returns:
            (B,) predictions in [0, 0.5]
        """
        B = len(samples)
        W = len(samples[0].window_weeks)
        dev = next(self.parameters()).device

        song_idxs = torch.tensor([s.song_idx for s in samples], dtype=torch.long, device=dev)

        seq_parts: list[Tensor] = []
        pad_cols: list[Tensor] = []

        for t in range(W):
            # Pad mask column for position t (may differ per sample / first_seen_week)
            pad_col = torch.tensor(
                [s.pad_mask[t] for s in samples], dtype=torch.bool, device=dev
            )
            pad_cols.append(pad_col)

            wk = samples[0].window_weeks[t]  # all samples in a week-grouped batch share window_weeks
            if wk in bank and not pad_col.all():
                # Vectorised advanced indexing: (B, hidden)
                emb = bank[wk][song_idxs]      # differentiable gather
                emb = emb.masked_fill(pad_col.unsqueeze(-1), 0.0)
            else:
                emb = torch.zeros(B, self.hidden, device=dev)
            seq_parts.append(emb)

        seq      = torch.stack(seq_parts, dim=1)          # (B, W, hidden)
        pad_mask = torch.stack(pad_cols, dim=1)            # (B, W) bool

        return self.head(seq, pad_mask)

    def count_params(self) -> int:
        """Return total number of trainable parameters.

        SAGEConv uses lazy initialization; call after at least one forward pass.
        """
        from torch.nn.parameter import UninitializedParameter
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter)
        )
