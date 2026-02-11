#!/usr/bin/env python
import argparse
import json
import os
import statistics
from pathlib import Path

import networkx as nx
import torch


def _latest_build_hash(build_graphs_dir: Path) -> str:
    candidates = [p for p in build_graphs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No build_graphs subdirs in {build_graphs_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


def _iter_graph_files(nx_root: Path):
    for graph_dir in sorted(nx_root.iterdir()):
        if not graph_dir.is_dir():
            continue
        for tw_file in sorted(graph_dir.iterdir()):
            if tw_file.is_file():
                yield graph_dir.name, tw_file


def _graph_metrics(G_multi: nx.MultiDiGraph):
    # Use a simple directed graph for structural checks
    G = nx.DiGraph(G_multi)
    U = nx.Graph(G)

    n = G.number_of_nodes()
    m = G.number_of_edges()

    roots = [n for n, d in G.in_degree() if d == 0]
    dag = nx.is_directed_acyclic_graph(G)

    scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
    wcc_sizes = [len(c) for c in nx.weakly_connected_components(G)]

    # Tree/forest checks on undirected components
    tree_components = 0
    tree_nodes = 0
    for comp in nx.connected_components(U):
        comp_nodes = len(comp)
        sub = U.subgraph(comp)
        if sub.number_of_edges() == comp_nodes - 1:
            tree_components += 1
            tree_nodes += comp_nodes

    metrics = {
        "nodes": n,
        "edges": m,
        "avg_degree": (2 * m / n) if n else 0.0,
        "roots": len(roots),
        "is_dag": dag,
        "largest_scc": max(scc_sizes) if scc_sizes else 0,
        "largest_wcc": max(wcc_sizes) if wcc_sizes else 0,
        "tree_component_ratio": (tree_components / len(wcc_sizes)) if wcc_sizes else 0.0,
        "tree_node_ratio": (tree_nodes / n) if n else 0.0,
        "is_forest": tree_components == len(wcc_sizes) if wcc_sizes else False,
    }

    if dag and n > 0:
        try:
            metrics["dag_longest_path"] = nx.dag_longest_path_length(G)
        except Exception:
            metrics["dag_longest_path"] = None
    else:
        metrics["dag_longest_path"] = None

    return metrics


def _aggregate(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze CADETS_E3 graph structure.")
    parser.add_argument("--dataset", default="CADETS_E3", help="Dataset name")
    parser.add_argument("--artifact-dir", default="/home/artifacts", help="Artifacts root")
    parser.add_argument("--build-hash", default=None, help="Specific build_graphs hash")
    parser.add_argument("--max-graphs", type=int, default=100, help="Max time-window graphs")
    parser.add_argument("--all", action="store_true", help="Process all time-window graphs")
    parser.add_argument("--out", default=None, help="Write full per-graph metrics JSON")
    args = parser.parse_args()

    build_graphs_dir = Path(args.artifact_dir) / "preprocessing" / args.dataset / "build_graphs"
    build_hash = args.build_hash or _latest_build_hash(build_graphs_dir)
    nx_root = build_graphs_dir / build_hash / "nx"

    if not nx_root.exists():
        raise FileNotFoundError(f"Missing nx folder: {nx_root}")

    metrics_list = []
    count = 0
    for graph_name, tw_file in _iter_graph_files(nx_root):
        G = torch.load(tw_file)
        if not isinstance(G, nx.MultiDiGraph):
            raise TypeError(f"Unexpected graph type {type(G)} in {tw_file}")
        m = _graph_metrics(G)
        m["graph"] = graph_name
        m["time_window"] = tw_file.name
        metrics_list.append(m)
        count += 1
        if not args.all and count >= args.max_graphs:
            break

    # Aggregate
    agg = {
        "dataset": args.dataset,
        "build_hash": build_hash,
        "graphs_analyzed": len(metrics_list),
        "nodes": _aggregate([m["nodes"] for m in metrics_list]),
        "edges": _aggregate([m["edges"] for m in metrics_list]),
        "avg_degree": _aggregate([m["avg_degree"] for m in metrics_list]),
        "roots": _aggregate([m["roots"] for m in metrics_list]),
        "dag_ratio": sum(1 for m in metrics_list if m["is_dag"]) / len(metrics_list)
        if metrics_list
        else 0.0,
        "largest_scc": _aggregate([m["largest_scc"] for m in metrics_list]),
        "largest_wcc": _aggregate([m["largest_wcc"] for m in metrics_list]),
        "tree_component_ratio": _aggregate([m["tree_component_ratio"] for m in metrics_list]),
        "tree_node_ratio": _aggregate([m["tree_node_ratio"] for m in metrics_list]),
        "forest_ratio": sum(1 for m in metrics_list if m["is_forest"]) / len(metrics_list)
        if metrics_list
        else 0.0,
        "dag_longest_path": _aggregate(
            [m["dag_longest_path"] for m in metrics_list if m["dag_longest_path"] is not None]
        ),
    }

    print(json.dumps(agg, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics_list, f, indent=2)


if __name__ == "__main__":
    main()
