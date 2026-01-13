import torch.nn as nn

from pidsmaker.encoders import SAGE


class CustomEncoder(nn.Module):
    """
    - Separate projections for source/destination node features at the edge level
    - Reindex/reshape to node-level features (N, d)
    - GraphSAGE over the whole graph to capture structure
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        graph_reindexer,
        activation,
        dropout: float,
        num_layers: int,
        device,
    ):
        super().__init__()

        self.src_proj = nn.Linear(in_dim, hid_dim)
        self.dst_proj = nn.Linear(in_dim, hid_dim)

        self.sage = SAGE(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.graph_reindexer = graph_reindexer

    def forward(self, x_src, x_dst, edge_index, **kwargs):
        # Edge-aligned projections (E, d)
        h_src = self.src_proj(x_src)
        h_dst = self.dst_proj(x_dst)

        # Convert edge-aligned (E, d) -> node-aligned (N, d)
        # Note: assumes graph_reindexer supports this API
        h_src_N, h_dst_N = self.graph_reindexer.node_features_reshape(
            edge_index, h_src, h_dst, x_is_tuple=True
        )
        h = h_src_N + h_dst_N  # (N, d)

        # GraphSAGE over nodes
        return self.sage(h, edge_index)
