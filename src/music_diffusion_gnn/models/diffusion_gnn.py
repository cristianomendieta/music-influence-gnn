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

    perm = torch.randperm(n_edges, device=store.edge_index.device)[:max_edges]
    snap[_COTRAJ_ET].edge_index = store.edge_index[:, perm]
    if "edge_attr" in store and store.edge_attr is not None:
        snap[_COTRAJ_ET].edge_attr = store.edge_attr[perm]
    if "first_seen_week" in store:
        snap[_COTRAJ_ET].first_seen_week = store.first_seen_week[perm]
    return snap


class MusicDiffusionGNN(nn.Module):
    """Heterogeneous temporal GNN for music popularity prediction.

    Architecture (R1, 2026-06-23 — lagged-popularity injection):
        encode_weeks: per week, concat the popularity bank ``pop_bank[w]`` (2 chart
            channels) onto the static music node features, then mask_until →
            HeteroSAGE → Z_music. Popularity thus *diffuses through the influence
            graph* (the central hypothesis), instead of structure-only embeddings.
        predict: gather the window sequence → GRU+MLP → residual Δ; the final
            prediction is ``ŷ = clamp(y_prev + Δ, 0, 0.5)`` where
            ``y_prev = pop_bank[w-1, song, chart]`` is exactly the naive-persistence
            value. The model only has to learn the structural *correction* to
            persistence (and, with zero-init head, starts by matching it).

    When ``pop_bank`` is ``None`` the model falls back to structure-only inputs
    with a zero persistence base (used by lightweight unit tests).

    The embedding bank is computed *once per distinct week per forward call*,
    cached intra-forward (not between epochs — weights change), and is fully
    part of the autograd graph.
    """

    def __init__(
        self,
        metadata: tuple,
        n_genre: int,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.2,
        genre_dim: int = 32,
        pop_bank: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        # Genres carry no descriptive attributes in the dataset, so they are
        # represented by a learnable embedding optimized end-to-end with the
        # rest of the model (registered here so it enters model.parameters()).
        self.genre_emb = nn.Embedding(n_genre, genre_dim)
        nn.init.normal_(self.genre_emb.weight, mean=0.0, std=0.1)
        self.encoder = HeteroSpatialEncoder(metadata, hidden=hidden, layers=layers, dropout=dropout)
        self.head = TemporalHead(hidden=hidden, dropout=dropout)
        # Per-week popularity (n_weeks, N_music, 2); registered as a buffer so it
        # follows .to(device) but is not a trainable parameter. None → fallback.
        if pop_bank is not None:
            self.register_buffer("pop_bank", pop_bank.float())
        else:
            self.pop_bank = None

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
            # Override the graph's (unused) static genre features with the
            # learnable embedding so genre identity is trained end-to-end.
            x_dict = dict(snap.x_dict)
            x_dict["genre"] = self.genre_emb.weight
            # R1: inject the week-w popularity (2 chart channels) as dynamic music
            # node features so it diffuses through the influence graph. w ≤ target-1,
            # so this is strictly past information (no leakage).
            if self.pop_bank is not None:
                x_dict["music"] = torch.cat(
                    [x_dict["music"], self.pop_bank[w]], dim=1
                )
            bank[w] = self.encoder(x_dict, snap.edge_index_dict)
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
            (B,) predictions in [0, 0.5] — ``clamp(y_prev + Δ, 0, 0.5)``
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

        delta = self.head(seq, pad_mask)                   # (B,) raw residual Δ

        # R1: anchor to naive persistence. y_prev = pop_bank[w-1, song, chart],
        # which equals persistence_predict (same 0.0 floor for gap weeks).
        if self.pop_bank is not None:
            prev_weeks = torch.tensor(
                [s.target_week - 1 for s in samples], dtype=torch.long, device=dev
            ).clamp_(min=0)
            chart_codes = torch.tensor(
                [s.chart for s in samples], dtype=torch.long, device=dev
            )
            y_prev = self.pop_bank[prev_weeks, song_idxs, chart_codes]  # (B,)
        else:
            y_prev = torch.zeros(B, device=dev)

        return (y_prev + delta).clamp(0.0, 0.5)

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
