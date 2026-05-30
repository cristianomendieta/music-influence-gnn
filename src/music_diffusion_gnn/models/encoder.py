"""HeteroGraphSAGE spatial encoder — embedding per music node from a graph snapshot."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroSpatialEncoder(nn.Module):
    """Multi-layer HeteroGraphSAGE encoder.

    Wraps one ``HeteroConv(SAGEConv)`` per message-passing layer. Each layer
    produces the same ``hidden``-dimensional output for every node type.
    Returns only the ``music`` node embeddings ``Z_music ∈ (N_music, hidden)``.
    """

    def __init__(
        self,
        metadata: tuple,
        hidden: int,
        layers: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.dropout = nn.Dropout(p=dropout)

        _, edge_types = metadata
        self.convs = nn.ModuleList()
        for _ in range(layers):
            conv = HeteroConv(
                {et: SAGEConv((-1, -1), hidden) for et in edge_types},
                aggr="sum",
            )
            self.convs.append(conv)

    def forward(self, x_dict: dict[str, Tensor], edge_index_dict: dict) -> Tensor:
        """Run all layers and return music node embeddings.

        Args:
            x_dict: node feature tensors keyed by node type
            edge_index_dict: edge indices keyed by (src, rel, dst) tuple

        Returns:
            Z_music of shape (N_music, hidden)
        """
        h = x_dict
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict)
            h = {k: v.relu() for k, v in h.items()}
            if i < len(self.convs) - 1:
                h = {k: self.dropout(v) for k, v in h.items()}
        return h["music"]
