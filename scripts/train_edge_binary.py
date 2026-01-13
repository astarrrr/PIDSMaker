#!/usr/bin/env python
"""Train a GraphSAGE/GAT edge classifier on OPTC time-window graphs.

This script treats each time-window graph as a unit (no cutting within windows),
labels edges as malicious if they touch a malicious node in that window, and
trains an edge MLP decoder with a GNN encoder.
"""

import argparse
import json
import math
import os
import random
from collections import Counter

import numpy as np
import torch
from torch import nn

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError as exc:
    raise SystemExit(
        "torch_geometric is required. Install it in the same environment and rerun."
    ) from exc

from pidsmaker.config.config import DATASET_DEFAULT_CONFIG
from pidsmaker.encoders.graph_attention import GraphAttentionEmbedding
from pidsmaker.encoders.sage import SAGE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_all_files_from_folders(base_dir: str, folders):
    paths = [
        os.path.abspath(os.path.join(base_dir, sub, f))
        for sub in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, sub)) and sub in folders
        for f in os.listdir(os.path.join(base_dir, sub))
    ]
    paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return paths


def find_latest_build_hash(build_root: str) -> str:
    if not os.path.isdir(build_root):
        raise FileNotFoundError(f"build_graphs root not found: {build_root}")
    candidates = []
    for name in os.listdir(build_root):
        path = os.path.join(build_root, name)
        done_file = os.path.join(path, "done.txt")
        if os.path.isdir(path) and os.path.exists(done_file):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No completed build_graphs runs found in {build_root}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.basename(candidates[0])


def load_graph(path: str):
    return torch.load(path, map_location="cpu")


def build_vocabs(paths, limit: int):
    node_types = Counter()
    edge_labels = Counter()
    for i, path in enumerate(paths):
        if limit and i >= limit:
            break
        g = load_graph(path)
        for _, data in g.nodes(data=True):
            node_types[data.get("node_type", "UNK")] += 1
        for _, _, data in g.edges(data=True):
            edge_labels[data.get("label", "UNK")] += 1
    node_vocab = sorted(node_types.keys())
    edge_vocab = sorted(edge_labels.keys())
    if "UNK" not in node_vocab:
        node_vocab.append("UNK")
    if "UNK" not in edge_vocab:
        edge_vocab.append("UNK")
    return node_vocab, edge_vocab


def graph_to_data(path, node_vocab, edge_vocab, malicious_nodes=None):
    g = load_graph(path)
    node_ids = list(g.nodes())
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    node_type_to_idx = {t: i for i, t in enumerate(node_vocab)}
    edge_label_to_idx = {t: i for i, t in enumerate(edge_vocab)}

    degrees = dict(g.degree())
    x = torch.zeros((len(node_ids), len(node_vocab) + 1), dtype=torch.float)
    for nid, data in g.nodes(data=True):
        nt = data.get("node_type", "UNK")
        x[node_index[nid], node_type_to_idx.get(nt, node_type_to_idx["UNK"])] = 1.0
        x[node_index[nid], -1] = math.log1p(degrees.get(nid, 0))

    edges = list(g.edges(data=True))
    edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
    edge_attr = torch.zeros((len(edges), len(edge_vocab) + 1), dtype=torch.float)
    edge_label = torch.zeros((len(edges),), dtype=torch.float)

    min_time = None
    for _, _, data in edges:
        t = data.get("time")
        if t is not None:
            min_time = t if min_time is None else min(min_time, t)

    mal_set = set()
    if malicious_nodes:
        mal_set = {str(nid) for nid in malicious_nodes}

    for i, (src, dst, data) in enumerate(edges):
        edge_index[0, i] = node_index[src]
        edge_index[1, i] = node_index[dst]
        label = data.get("label", "UNK")
        edge_attr[i, edge_label_to_idx.get(label, edge_label_to_idx["UNK"])] = 1.0
        if data.get("time") is not None and min_time is not None:
            edge_attr[i, -1] = float(data["time"] - min_time) / 1e9
        if mal_set and (str(src) in mal_set or str(dst) in mal_set):
            edge_label[i] = 1.0

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_label)


class EdgeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_src, z_dst, edge_attr):
        feats = torch.cat([z_src, z_dst, edge_attr], dim=1)
        return self.net(feats).squeeze(1)


def count_labels(dataset):
    pos = 0
    total = 0
    for data in dataset:
        pos += int(data.y.sum().item())
        total += data.y.numel()
    return pos, total - pos


def compute_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_seed(args, train_set, val_set, test_set, edge_attr_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = train_set[0].x.shape[1]
    out_dim = args.out_dim

    if args.encoder == "sage":
        encoder = SAGE(
            in_dim=in_dim,
            hid_dim=args.hidden_dim,
            out_dim=out_dim,
            activation=nn.ReLU(),
            dropout=args.dropout,
            num_layers=args.num_layers,
        )
    else:
        encoder = GraphAttentionEmbedding(
            in_dim=in_dim,
            hid_dim=args.hidden_dim,
            out_dim=out_dim,
            edge_dim=edge_attr_dim,
            dropout=args.dropout,
            activation=nn.ReLU(),
            num_heads=args.num_heads,
            concat=True,
            num_layers=args.num_layers,
        )

    decoder = EdgeMLP(in_dim=out_dim * 2 + edge_attr_dim, hidden_dim=args.edge_hidden)
    encoder.to(device)
    decoder.to(device)

    pos, neg = count_labels(train_set)
    if pos == 0:
        pos_weight = torch.tensor([1.0], device=device)
        print("Warning: no positive edges in training set; using pos_weight=1.0")
    else:
        pos_weight = torch.tensor([neg / pos], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            enc_out = encoder(batch.x, batch.edge_index, edge_feats=batch.edge_attr)
            h = enc_out["h"]
            logits = decoder(h[batch.edge_index[0]], h[batch.edge_index[1]], batch.edge_attr)
            loss = criterion(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                batch = batch.to(device)
                enc_out = encoder(batch.x, batch.edge_index, edge_feats=batch.edge_attr)
                h = enc_out["h"]
                logits = decoder(h[batch.edge_index[0]], h[batch.edge_index[1]], batch.edge_attr)
                val_losses.append(criterion(logits, batch.y).item())
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.cpu() for k, v in decoder.state_dict().items()},
            }

        print(f"epoch {epoch:02d} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}")

    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    encoder.eval()
    decoder.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            enc_out = encoder(batch.x, batch.edge_index, edge_feats=batch.edge_attr)
            h = enc_out["h"]
            logits = decoder(h[batch.edge_index[0]], h[batch.edge_index[1]], batch.edge_attr)
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    metrics = compute_metrics(labels.long(), preds)
    metrics["pos_rate"] = float(labels.float().mean().item())
    return encoder, decoder, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="optc_h051")
    parser.add_argument("--artifact_dir", default="/home/artifacts")
    parser.add_argument("--build_graphs_hash", default="")
    parser.add_argument("--output_dir", default="/home/pids/outputs/edge_binary")
    parser.add_argument("--encoder", choices=["sage", "gat"], default="sage")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--edge_hidden", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--vocab_scan_limit", type=int, default=200)
    parser.add_argument("--max_windows_per_split", type=int, default=0)
    args = parser.parse_args()

    if args.dataset not in DATASET_DEFAULT_CONFIG:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    ds_cfg = DATASET_DEFAULT_CONFIG[args.dataset]

    build_root = os.path.join(args.artifact_dir, "preprocessing", args.dataset, "build_graphs")
    build_hash = args.build_graphs_hash or find_latest_build_hash(build_root)
    nx_root = os.path.join(build_root, build_hash, "nx")

    train_paths = get_all_files_from_folders(nx_root, ds_cfg["train_files"])
    val_paths = get_all_files_from_folders(nx_root, ds_cfg["val_files"])
    test_paths = get_all_files_from_folders(nx_root, ds_cfg["test_files"])

    if args.max_windows_per_split:
        train_paths = train_paths[: args.max_windows_per_split]
        val_paths = val_paths[: args.max_windows_per_split]
        test_paths = test_paths[: args.max_windows_per_split]

    all_paths_for_vocab = train_paths + val_paths + test_paths
    node_vocab, edge_vocab = build_vocabs(all_paths_for_vocab, args.vocab_scan_limit)

    tw_labels_path = os.path.join(build_root, build_hash, "tw_labels", "tw_to_malicious_nodes.pkl")
    tw_to_malicious = torch.load(tw_labels_path, map_location="cpu")
    test_idx_to_mal = {}
    for idx, nodes in tw_to_malicious.items():
        if isinstance(nodes, dict):
            test_idx_to_mal[int(idx)] = list(nodes.keys())
        else:
            test_idx_to_mal[int(idx)] = list(nodes)

    train_set = [graph_to_data(p, node_vocab, edge_vocab) for p in train_paths]
    val_set = [graph_to_data(p, node_vocab, edge_vocab) for p in val_paths]
    test_set = []
    for i, p in enumerate(test_paths):
        malicious_nodes = test_idx_to_mal.get(i)
        test_set.append(graph_to_data(p, node_vocab, edge_vocab, malicious_nodes=malicious_nodes))

    edge_attr_dim = len(edge_vocab) + 1
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    per_run_metrics = []

    for seed in seeds:
        set_seed(seed)
        encoder, decoder, metrics = train_one_seed(
            args, train_set, val_set, test_set, edge_attr_dim
        )
        run_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)
        torch.save(encoder.state_dict(), os.path.join(run_dir, "encoder.pt"))
        torch.save(decoder.state_dict(), os.path.join(run_dir, "decoder.pt"))
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        per_run_metrics.append(metrics)

    summary = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        vals = [m[key] for m in per_run_metrics]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "artifact_dir": args.artifact_dir,
                "build_graphs_hash": build_hash,
                "train_files": ds_cfg["train_files"],
                "val_files": ds_cfg["val_files"],
                "test_files": ds_cfg["test_files"],
                "encoder": args.encoder,
                "num_layers": args.num_layers,
                "hidden_dim": args.hidden_dim,
                "out_dim": args.out_dim,
                "edge_hidden": args.edge_hidden,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seeds": seeds,
            },
            f,
            indent=2,
        )

    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
