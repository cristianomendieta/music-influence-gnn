"""Per-relation GATv2 spatial encoder with a root term (ADR-0006)."""
from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, Linear
from torch_geometric.nn.module_dict import ModuleDict

# The only relation with causal edge attributes ([chart, log1p(age)], see
# graph/edge_features.py). Every other relation attends by content only.
_EDGE_DIM = {("music", "cotrajectory", "music"): 2}


class HeteroAttentionEncoder(nn.Module):
    """Multi-layer heterogeneous encoder with one ``GATv2Conv`` per relation.

    Each layer computes, for every destination node type ``t``,
    ``ReLU(W_root,t · x_t + Σ_r GATv2_r(x_src, x_t))`` over the relations ``r``
    that arrive at ``t``. ``add_self_loops=False``: the root term plays the
    role of the self loop. Without edges the encoder reduces to the root term,
    which is the same function class as ``HeteroSpatialEncoder`` on an
    edgeless graph (the sem_grafo control, ADR-0006). Returns only
    ``Z_music ∈ (N_music, hidden)``.
    """

    def __init__(
        self,
        metadata: tuple,
        hidden: int,
        layers: int,
        heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} is not divisible by heads={heads}")
        self.hidden = hidden
        self.layers = layers
        self.dropout = nn.Dropout(p=dropout)

        _, edge_types = metadata
        self.edge_types = list(edge_types)
        dst_types = list(dict.fromkeys(et[2] for et in self.edge_types))
        self.roots = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.roots.append(nn.ModuleDict({t: Linear(-1, hidden) for t in dst_types}))
            self.convs.append(ModuleDict({
                et: GATv2Conv(
                    (-1, -1),
                    hidden // heads,
                    heads=heads,
                    concat=True,
                    dropout=attn_dropout,
                    add_self_loops=False,
                    edge_dim=_EDGE_DIM.get(et),
                )
                for et in self.edge_types
            }))

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict,
        edge_attr_dict: dict | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, list[dict[tuple, tuple[Tensor, Tensor]]]]:
        """Run all layers and return music node embeddings.

        Args:
            x_dict: node feature tensors keyed by node type
            edge_index_dict: edge indices keyed by (src, rel, dst) tuple
            edge_attr_dict: causal edge features from ``causal_edge_features``;
                read only for the relations in ``_EDGE_DIM``
            return_attention: also return the attention weights

        Returns:
            Z_music of shape (N_music, hidden). With ``return_attention=True``,
            ``(Z_music, attention)`` where ``attention[layer][edge_type]`` is
            ``(edge_index, alpha)`` and ``alpha`` is ``(E, heads)``. Parallel
            edges stay separate entries, each with its own weight.
        """
        edge_attr_dict = edge_attr_dict or {}
        attention: list[dict[tuple, tuple[Tensor, Tensor]]] = []
        h = x_dict
        for i in range(self.layers):
            out = {t: root(h[t]) for t, root in self.roots[i].items()}
            attention.append({})
            for et in self.edge_types:
                src, _, dst = et
                edge_index = edge_index_dict[et]
                edge_attr = edge_attr_dict[et] if et in _EDGE_DIM else None
                conv = self.convs[i][et]
                if return_attention:
                    msg, attention[i][et] = conv(
                        (h[src], h[dst]), edge_index, edge_attr=edge_attr,
                        return_attention_weights=True,
                    )
                else:
                    msg = conv((h[src], h[dst]), edge_index, edge_attr=edge_attr)
                # An empty relation still runs (its lazy weights materialise) but
                # contributes nothing, not even its bias (AC7).
                if edge_index.shape[1] > 0:
                    out[dst] = out[dst] + msg
            h = {k: v.relu() for k, v in out.items()}
            if i < self.layers - 1:
                h = {k: self.dropout(v) for k, v in h.items()}
        if return_attention:
            return h["music"], attention
        return h["music"]

    def count_params(self) -> int:
        """Return the number of trainable parameters already initialised.

        The root ``Linear`` and the ``GATv2Conv`` projections are lazy; call
        after at least one forward pass to get the full count.
        """
        from torch.nn.parameter import UninitializedParameter
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter)
        )
