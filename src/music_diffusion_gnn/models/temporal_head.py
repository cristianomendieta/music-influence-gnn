"""GRU temporal head: sequence of embeddings → residual correction Δ (R1)."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class TemporalHead(nn.Module):
    """GRU (1 layer) over a window of music embeddings + MLP head.

    **R1 (2026-06-23):** the head now outputs a *raw residual correction* ``Δ``
    (no final sigmoid). The caller (:meth:`MusicDiffusionGNN.predict`) forms the
    final prediction as ``ŷ = clamp(y_prev + Δ, 0, 0.5)``, anchoring the model to
    the naive-persistence baseline. The last ``Linear`` is **zero-initialised**
    so that ``Δ = 0`` at step 0 → the untrained model reproduces persistence and
    only has to learn the structural correction.

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
        # Zero-init the output layer → Δ starts at 0 → ŷ starts at persistence.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, seq: Tensor, pad_mask: Tensor) -> Tensor:
        """Predict the residual correction Δ from an embedding sequence.

        Args:
            seq: (B, W, hidden) — embedding for each look-back week
            pad_mask: (B, W) bool — True where the step is padding

        Returns:
            (B,) raw residual Δ ∈ ℝ (combined with y_prev + clamp by the caller)
        """
        # Zero out padded positions so they don't influence hidden state
        seq = seq.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        _, h_n = self.gru(seq)  # h_n: (1, B, hidden)
        h = h_n.squeeze(0)      # (B, hidden)
        return self.mlp(h).squeeze(-1)  # (B,) raw Δ
