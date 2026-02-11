from types import SimpleNamespace

import torch
import torch.nn as nn

from pidsmaker.encoders.hypformer.hypformer import HypFormer


class HypformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        dropout,
        trans_num_layers,
        trans_num_heads,
        trans_use_bn,
        trans_use_residual,
        trans_use_weight,
        trans_use_act,
        k_in,
        k_out,
        decoder_type,
        add_positional_encoding,
        attention_type,
        power_k,
        trans_heads_concat,
        use_edge_index_mask,
        graph_reindexer,
        x_is_tuple,
        device,
    ):
        super().__init__()
        self.graph_reindexer = graph_reindexer
        self.x_is_tuple = x_is_tuple
        self.use_edge_index_mask = use_edge_index_mask
        self.attention_type = attention_type

        args = SimpleNamespace(
            k_in=k_in,
            k_out=k_out,
            decoder_type=decoder_type,
            device=device,
            add_positional_encoding=add_positional_encoding,
            attention_type=attention_type,
            power_k=power_k,
            trans_heads_concat=trans_heads_concat,
        )

        self.model = HypFormer(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            out_channels=out_dim,
            trans_num_layers=trans_num_layers,
            trans_num_heads=trans_num_heads,
            trans_dropout=dropout,
            trans_use_bn=trans_use_bn,
            trans_use_residual=trans_use_residual,
            trans_use_weight=trans_use_weight,
            trans_use_act=trans_use_act,
            args=args,
        )

    @staticmethod
    def _build_attention_mask(edge_index, num_nodes, device):
        mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        if edge_index is not None and edge_index.numel() > 0:
            mask[edge_index[0], edge_index[1]] = True
        mask.fill_diagonal_(True)
        return mask

    def _resolve_x(self, x, x_src, x_dst, edge_index):
        if x is None:
            if edge_index is None or x_src is None or x_dst is None:
                raise ValueError("HypformerEncoder requires x or edge_index/x_src/x_dst.")
            x_src_nodes, x_dst_nodes = self.graph_reindexer.node_features_reshape(
                edge_index, x_src, x_dst, x_is_tuple=True
            )
            x = 0.5 * (x_src_nodes + x_dst_nodes)
        elif isinstance(x, (tuple, list)):
            x = 0.5 * (x[0] + x[1])
        return x

    def forward(self, x=None, x_src=None, x_dst=None, edge_index=None, **kwargs):
        x = self._resolve_x(x, x_src, x_dst, edge_index)
        attention_mask = None
        if self.use_edge_index_mask:
            if self.attention_type != "full":
                raise ValueError("Edge-index attention masks require attention_type='full'.")
            if edge_index is None:
                raise ValueError("Edge-index attention masks require edge_index.")
            attention_mask = self._build_attention_mask(edge_index, x.size(0), x.device)
        h = self.model(x, attention_mask=attention_mask)
        return {"h": h}
