import os

import torch
from torch_scatter import scatter

from pidsmaker.utils.data_utils import load_all_datasets
from pidsmaker.utils.utils import get_device, log, log_start, set_seed


def _get_context_cfg(cfg):
    context_k = getattr(cfg.detection.graph_preprocessing, "context_k", 0) or 0
    context_agg = getattr(cfg.detection.graph_preprocessing, "context_agg", "mean") or "mean"
    return int(context_k), context_agg


def _parse_selected_node_feats(cfg):
    selected_node_feats = cfg.detection.graph_preprocessing.node_features
    only_type = cfg.featurization.feat_training.used_method.strip() == "only_type"
    only_ones = cfg.featurization.feat_training.used_method.strip() == "only_ones"
    if only_type:
        return ["node_type"]
    if only_ones:
        return ["only_ones"]
    return list(map(lambda x: x.strip(), selected_node_feats.replace("-", ",").split(",")))


def _get_node_emb_slice(cfg, x_dim):
    emb_dim = cfg.featurization.feat_training.emb_dim
    if emb_dim is None or emb_dim == 0:
        return None

    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types
    field_to_size = {
        "node_emb": emb_dim,
        "node_type": node_type_dim,
        "edges_distribution": edge_type_dim * 2,
        "only_ones": node_type_dim,
    }

    offset = 0
    emb_start = None
    emb_end = None
    for feat in _parse_selected_node_feats(cfg):
        size = field_to_size.get(feat)
        if size is None:
            raise ValueError(f"Node feature {feat} is invalid.")
        if feat == "node_emb":
            emb_start = offset
            emb_end = offset + size
        offset += size

    if emb_start is None or emb_end is None:
        return None
    if offset == x_dim:
        return emb_start, emb_end, False
    if offset + emb_dim == x_dim:
        return emb_start, emb_end, True
    log(
        f"Warning: x_src dim {x_dim} doesn't match expected {offset} or {offset + emb_dim}; "
        "skipping context.",
    )
    return None


def _build_context_embeddings(graph, context_k, emb_slice):
    src = graph.src
    dst = graph.dst
    t = graph.t
    x_src = graph.x_src
    x_dst = graph.x_dst

    emb_start, emb_end, _ = emb_slice
    src_emb = x_src[:, emb_start:emb_end]
    dst_emb = x_dst[:, emb_start:emb_end]
    emb_dim = src_emb.shape[1]

    max_node = int(torch.cat([src, dst]).max().item()) + 1
    zeros = torch.zeros((emb_dim,), dtype=src_emb.dtype, device=src_emb.device)

    neighbor_lists = {}
    order = torch.argsort(t)
    for idx in order.tolist():
        u = int(src[idx])
        v = int(dst[idx])

        u_list = neighbor_lists.setdefault(u, [])
        u_list.append(dst_emb[idx])
        if len(u_list) > context_k:
            u_list.pop(0)

        v_list = neighbor_lists.setdefault(v, [])
        v_list.append(src_emb[idx])
        if len(v_list) > context_k:
            v_list.pop(0)

    context_by_node = torch.zeros((max_node, emb_dim), dtype=src_emb.dtype, device=src_emb.device)
    for node_id, emb_list in neighbor_lists.items():
        if len(emb_list) == 0:
            context_by_node[node_id] = zeros
        else:
            context_by_node[node_id] = torch.stack(emb_list, dim=0).mean(dim=0)

    context_src = context_by_node[src]
    context_dst = context_by_node[dst]
    return context_src, context_dst


def _refresh_node_features(graph, x_is_tuple):
    if not hasattr(graph, "edge_index"):
        return
    edge_index = graph.edge_index
    max_num_node = int(edge_index.max().item()) + 1
    feature_dim = graph.x_src.size(1)

    output = torch.zeros((max_num_node, feature_dim), device=graph.x_src.device)
    if x_is_tuple:
        scatter(graph.x_src, edge_index[0], out=output, dim=0, reduce="mean")
        x_src_nodes = output.clone()
        output.zero_()
        scatter(graph.x_dst, edge_index[1], out=output, dim=0, reduce="mean")
        x_dst_nodes = output.clone()
        graph.x = (x_src_nodes, x_dst_nodes)
    else:
        scatter(
            torch.cat([graph.x_src, graph.x_dst]),
            torch.cat([edge_index[0], edge_index[1]]),
            out=output,
            dim=0,
            reduce="mean",
        )
        graph.x = output


def _apply_context_embeddings(datasets, cfg):
    context_k, context_agg = _get_context_cfg(cfg)
    if context_k <= 0:
        return datasets
    if context_agg != "mean":
        raise ValueError(f"Invalid context_agg {context_agg}")
    x_is_tuple = cfg.detection.gnn_training.encoder.x_is_tuple

    logged = False
    for dataset in datasets:
        emb_slices = []
        for graph in dataset:
            if getattr(graph, "_context_emb_added", False):
                emb_slices.append((graph, None))
                continue

            emb_slice = _get_node_emb_slice(cfg, graph.x_src.shape[1])
            if emb_slice is None:
                log("Warning: skipping context embeddings due to unexpected x_src dimensions.")
                return datasets
            emb_slices.append((graph, emb_slice))

        for graph, emb_slice in emb_slices:
            if emb_slice is None:
                continue
            emb_start, emb_end, already_has_context = emb_slice
            if already_has_context:
                graph._context_emb_added = True
                continue

            old_dim = graph.x_src.shape[1]
            context_src, context_dst = _build_context_embeddings(graph, context_k, emb_slice)

            graph.x_src = torch.cat(
                [graph.x_src[:, :emb_end], context_src, graph.x_src[:, emb_end:]], dim=-1
            )
            graph.x_dst = torch.cat(
                [graph.x_dst[:, :emb_end], context_dst, graph.x_dst[:, emb_end:]], dim=-1
            )
            if hasattr(graph, "x"):
                _refresh_node_features(graph, x_is_tuple)
            graph._context_emb_added = True

            if not logged:
                log(
                    f"Applied context embeddings (k={context_k}, agg={context_agg}); "
                    f"x_dim: {old_dim} -> {graph.x_src.shape[1]}"
                )
                logged = True

    return datasets


def get_preprocessed_graphs(cfg):
    if cfg.detection.graph_preprocessing.save_on_disk:
        log("Loading preprocessed graphs...")
        out_dir = cfg.detection.graph_preprocessing._preprocessed_graphs_dir
        out_file = os.path.join(out_dir, "torch_graphs.pkl")
        train_data, val_data, test_data, max_node_num = torch.load(out_file)

    else:
        log("Computing graphs...")
        device = get_device(cfg)
        train_data, val_data, test_data, max_node_num = load_all_datasets(cfg, device)

    train_data = _apply_context_embeddings(train_data, cfg)
    val_data = _apply_context_embeddings(val_data, cfg)
    test_data = _apply_context_embeddings(test_data, cfg)

    return train_data, val_data, test_data, max_node_num


def main(cfg):
    set_seed(cfg)
    log_start(__file__)

    if cfg.detection.graph_preprocessing.save_on_disk:
        device = get_device(cfg)
        train_data, val_data, test_data, max_node_num = load_all_datasets(cfg, device)
        train_data = _apply_context_embeddings(train_data, cfg)
        val_data = _apply_context_embeddings(val_data, cfg)
        test_data = _apply_context_embeddings(test_data, cfg)

        out_dir = cfg.detection.graph_preprocessing._preprocessed_graphs_dir
        out_file = os.path.join(out_dir, "torch_graphs.pkl")
        os.makedirs(out_dir, exist_ok=True)
        log(f"Saving preprocessed graphs to {out_file}...")
        torch.save((train_data, val_data, test_data, max_node_num), out_file)

    else:
        log("Not saving to disk, skipping this task.")
