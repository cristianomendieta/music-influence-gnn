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
    ) -> dict[int, Tensor]:
        """Compute music embeddings for each distinct week (one forward per week).

        Args:
            g: the full ``hetero_full`` graph
            weeks: list of week indices to encode (duplicates encoded once)

        Returns:
            ``{week: Z_music}`` where each Z_music ∈ (N_music, hidden)
        """
        bank: dict[int, Tensor] = {}
        for w in set(weeks):
            snap = mask_until(g, w)
            bank[w] = self.encoder(snap.x_dict, snap.edge_index_dict)
        return bank

    def predict(
        self,
        bank: dict[int, Tensor],
        samples: list[Sample],
    ) -> Tensor:
        """Assemble window sequences and predict rank scores.

        Args:
            bank: ``{week: Z_music}`` from ``encode_weeks``
            samples: list of Sample objects (same batch)

        Returns:
            (B,) predictions in [0, 0.5]
        """
        B = len(samples)
        W = len(samples[0].window_weeks)

        # Build (B, W, hidden) sequence tensor and (B, W) pad mask
        seq = torch.zeros(B, W, self.hidden, device=next(self.parameters()).device)
        pad_mask = torch.zeros(B, W, dtype=torch.bool, device=seq.device)

        for b, samp in enumerate(samples):
            for t, (wk, is_pad) in enumerate(zip(samp.window_weeks, samp.pad_mask)):
                pad_mask[b, t] = is_pad
                if not is_pad:
                    seq[b, t] = bank[wk][samp.song_idx]

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
