#!/usr/bin/env python
"""Train a benign-only autoencoder and use reconstruction error for anomaly detection.

This script builds simple graph-level features from precomputed NetworkX time-window graphs
and trains an encoder/decoder on benign windows only. At evaluation time, it thresholds
reconstruction error to classify benign vs anomalous windows.
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
from torch.utils.data import DataLoader, TensorDataset

from pidsmaker.config.config import DATASET_DEFAULT_CONFIG


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def get_all_files_from_folders(base_dir: str, folders):
    paths = [
        os.path.abspath(os.path.join(base_dir, sub, f))
        for sub in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, sub)) and sub in folders
        for f in os.listdir(os.path.join(base_dir, sub))
    ]
    paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return paths


def load_graph(path: str):
    return torch.load(path, map_location="cpu")


def build_vocab(paths, limit: int):
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


def graph_to_features(path: str, node_vocab, edge_vocab):
    g = load_graph(path)
    node_counts = Counter([d.get("node_type", "UNK") for _, d in g.nodes(data=True)])
    edge_counts = Counter([d.get("label", "UNK") for _, _, d in g.edges(data=True)])
    num_nodes = max(g.number_of_nodes(), 1)
    num_edges = max(g.number_of_edges(), 1)

    feats = []
    for nt in node_vocab:
        feats.append(node_counts.get(nt, 0) / num_nodes)
    for et in edge_vocab:
        feats.append(edge_counts.get(et, 0) / num_edges)
    feats.append(math.log1p(num_nodes))
    feats.append(math.log1p(num_edges))
    return np.array(feats, dtype=np.float32)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def mse_per_sample(x, recon):
    return ((x - recon) ** 2).mean(dim=1)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="optc_h051")
    parser.add_argument("--artifact_dir", default="/home/artifacts")
    parser.add_argument("--build_graphs_hash", default="")
    parser.add_argument("--output_dir", default="/home/pids/outputs/benign_autoencoder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--threshold_std", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_scan_limit", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.dataset not in DATASET_DEFAULT_CONFIG:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    ds_cfg = DATASET_DEFAULT_CONFIG[args.dataset]

    build_root = os.path.join(args.artifact_dir, "preprocessing", args.dataset, "build_graphs")
    build_hash = args.build_graphs_hash or find_latest_build_hash(build_root)
    nx_root = os.path.join(build_root, build_hash, "nx")

    train_paths = get_all_files_from_folders(nx_root, ds_cfg["train_files"])
    val_paths = get_all_files_from_folders(nx_root, ds_cfg["val_files"])
    test_paths = get_all_files_from_folders(nx_root, ds_cfg["test_files"])

    if not train_paths or not val_paths or not test_paths:
        raise RuntimeError("Missing train/val/test graph files. Check preprocessing output.")

    node_vocab, edge_vocab = build_vocab(train_paths, args.vocab_scan_limit)

    def build_matrix(paths):
        feats = [graph_to_features(p, node_vocab, edge_vocab) for p in paths]
        return np.stack(feats, axis=0)

    x_train = build_matrix(train_paths)
    x_val = build_matrix(val_paths)
    x_test = build_matrix(test_paths)

    # Labels: train/val are benign; test labels from malicious time windows.
    y_train = np.zeros(len(x_train), dtype=np.int64)
    y_val = np.zeros(len(x_val), dtype=np.int64)

    tw_labels_path = os.path.join(build_root, build_hash, "tw_labels", "tw_to_malicious_nodes.pkl")
    tw_to_malicious = torch.load(tw_labels_path, map_location="cpu")
    y_test = np.zeros(len(x_test), dtype=np.int64)
    for idx in range(len(x_test)):
        if idx in tw_to_malicious:
            y_test[idx] = 1

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_tensor = torch.from_numpy(x_val)
    test_tensor = torch.from_numpy(x_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(x_train.shape[1], args.hidden_dim, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor.to(device))
            val_loss = loss_fn(val_recon, val_tensor.to(device)).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_recon = model(val_tensor.to(device)).cpu()
        test_recon = model(test_tensor.to(device)).cpu()

    val_scores = mse_per_sample(val_tensor, val_recon).numpy()
    test_scores = mse_per_sample(test_tensor, test_recon).numpy()

    threshold = float(val_scores.mean() + args.threshold_std * val_scores.std())
    y_pred = (test_scores > threshold).astype(np.int64)

    metrics = compute_metrics(y_test, y_pred)
    metrics.update(
        {
            "threshold": threshold,
            "val_score_mean": float(val_scores.mean()),
            "val_score_std": float(val_scores.std()),
            "test_score_mean": float(test_scores.mean()),
            "test_score_std": float(test_scores.std()),
        }
    )

    os.makedirs(args.output_dir, exist_ok=True)
    encoder_path = os.path.join(args.output_dir, "encoder.pt")
    decoder_path = os.path.join(args.output_dir, "decoder.pt")

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    with open(os.path.join(args.output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"node_types": node_vocab, "edge_labels": edge_vocab}, f, indent=2)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "artifact_dir": args.artifact_dir,
                "build_graphs_hash": build_hash,
                "train_files": ds_cfg["train_files"],
                "val_files": ds_cfg["val_files"],
                "test_files": ds_cfg["test_files"],
            },
            f,
            indent=2,
        )

    print("Saved encoder:", encoder_path)
    print("Saved decoder:", decoder_path)
    print("Saved metrics:", os.path.join(args.output_dir, "metrics.json"))


if __name__ == "__main__":
    main()
