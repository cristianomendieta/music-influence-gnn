"""GRU temporal head: sequence of embeddings → ŷ ∈ [0, 0.5]."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TemporalHead(nn.Module):
    """GRU (1 layer) over a window of music embeddings + MLP head.

    Output is constrained to [0, 0.5] via ``0.5 * sigmoid(.)``.
    Padding is handled by zeroing out padded time steps before the GRU.
    """

    def __init__(self, hidden: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq: Tensor, pad_mask: Tensor) -> Tensor:
        """Predict rank score from embedding sequence.

        Args:
            seq: (B, W, hidden) — embedding for each look-back week
            pad_mask: (B, W) bool — True where the step is padding

        Returns:
            (B,) predictions in [0, 0.5]
        """
        # Zero out padded positions so they don't influence hidden state
        seq = seq.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        _, h_n = self.gru(seq)  # h_n: (1, B, hidden)
        h = h_n.squeeze(0)      # (B, hidden)
        out = self.mlp(h).squeeze(-1)  # (B,)
        return 0.5 * torch.sigmoid(out)
