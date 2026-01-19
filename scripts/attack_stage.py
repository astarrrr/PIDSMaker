import argparse
import os
from collections import defaultdict

import pandas as pd

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    datetime_to_ns_time_US_handle_nano,
    get_threshold,
    reduce_losses_to_score,
)
from pidsmaker.utils.labelling import get_GP_of_each_attack
from pidsmaker.utils.utils import datetime_to_ns_time_US, listdir_sorted


def _load_tw_ranges(test_tw_path):
    filelist = listdir_sorted(test_tw_path)
    tw_ranges = []
    for filename in filelist:
        name = os.path.splitext(filename)[0]
        start_str, end_str = name.split("~")
        start_ns = datetime_to_ns_time_US_handle_nano(start_str)
        end_ns = datetime_to_ns_time_US_handle_nano(end_str)
        tw_ranges.append((start_ns, end_ns))
    return filelist, tw_ranges


def _attack_tw_indices(attack_start_ns, attack_end_ns, tw_ranges):
    indices = []
    for i, (tw_start, tw_end) in enumerate(tw_ranges):
        if (tw_end >= attack_start_ns) and (tw_start <= attack_end_ns):
            indices.append(i)
    return indices


def _stage_for_tw(tw_idx, attack_indices):
    if not attack_indices:
        return "unknown"
    if tw_idx not in attack_indices:
        return "outside"
    pos = attack_indices.index(tw_idx)
    total = len(attack_indices)
    if total == 1:
        return "middle"
    if pos < total / 3:
        return "start"
    if pos < (2 * total) / 3:
        return "middle"
    return "end"


def _compute_node_scores(test_tw_path, use_dst_node_loss, threshold_method):
    node_to_losses = defaultdict(list)
    node_to_max_loss = defaultdict(float)
    node_to_max_loss_tw = {}

    filelist = listdir_sorted(test_tw_path)
    for tw, filename in enumerate(filelist):
        fpath = os.path.join(test_tw_path, filename)
        df = pd.read_csv(fpath).to_dict(orient="records")
        for line in df:
            srcnode = line["srcnode"]
            dstnode = line["dstnode"]
            loss = float(line["loss"])

            node_to_losses[srcnode].append(loss)
            if loss > node_to_max_loss[srcnode]:
                node_to_max_loss[srcnode] = loss
                node_to_max_loss_tw[srcnode] = tw

            if use_dst_node_loss:
                node_to_losses[dstnode].append(loss)
                if loss > node_to_max_loss[dstnode]:
                    node_to_max_loss[dstnode] = loss
                    node_to_max_loss_tw[dstnode] = tw

    node_to_scores = {}
    for node_id, losses in node_to_losses.items():
        node_to_scores[node_id] = reduce_losses_to_score(losses, threshold_method)

    return node_to_scores, node_to_max_loss_tw


def main():
    parser = argparse.ArgumentParser(
        description="Label detected malicious nodes as start/middle/end of each attack."
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default="",
        help="Epoch directory to use (e.g., epoch_11). Defaults to last available.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="attack_stage.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--gnn_training_path",
        type=str,
        default="",
        help="Override gnn_training task path (folder that contains edge_losses/).",
    )
    parser.add_argument(
        "--edge_losses_dir",
        type=str,
        default="",
        help="Override edge_losses directory (contains val/ and test/).",
    )

    args, unknown = get_runtime_required_args(return_unknown_args=True, args=None)
    args = parser.parse_args(unknown, namespace=args)
    cfg = get_yml_cfg(args)

    if args.edge_losses_dir:
        edge_losses_dir = args.edge_losses_dir
    elif args.gnn_training_path:
        edge_losses_dir = os.path.join(args.gnn_training_path, "edge_losses")
    else:
        edge_losses_dir = cfg.detection.gnn_training._edge_losses_dir

    test_losses_dir = os.path.join(edge_losses_dir, "test")
    val_losses_dir = os.path.join(edge_losses_dir, "val")
    epochs = listdir_sorted(test_losses_dir) if os.path.exists(test_losses_dir) else []
    if not epochs:
        raise FileNotFoundError(f"No test loss files found in {test_losses_dir}")

    epoch_dir = args.epoch if args.epoch else epochs[-1]
    if epoch_dir not in epochs and not epoch_dir.startswith("model_epoch_"):
        candidate = f"model_epoch_{epoch_dir.replace('epoch_', '')}"
        if candidate in epochs:
            epoch_dir = candidate
    test_tw_path = os.path.join(test_losses_dir, epoch_dir)
    val_tw_path = os.path.join(val_losses_dir, epoch_dir)

    filelist, tw_ranges = _load_tw_ranges(test_tw_path)

    attack_to_nids = get_GP_of_each_attack(cfg)
    attack_windows = {}
    for attack_id, payload in attack_to_nids.items():
        attack_start_ns, attack_end_ns = payload["time_range"]
        attack_windows[attack_id] = _attack_tw_indices(
            attack_start_ns, attack_end_ns, tw_ranges
        )

    threshold_method = cfg.detection.evaluation.node_evaluation.threshold_method
    if threshold_method == "magic":
        thr = get_threshold(test_tw_path, threshold_method)
    else:
        if not os.path.exists(val_tw_path):
            print(f"Warning: missing {val_tw_path}, falling back to test split for threshold.")
            thr = get_threshold(test_tw_path, threshold_method)
        else:
            thr = get_threshold(val_tw_path, threshold_method)

    use_dst_node_loss = cfg.detection.evaluation.node_evaluation.use_dst_node_loss
    node_to_scores, node_to_max_loss_tw = _compute_node_scores(
        test_tw_path, use_dst_node_loss, threshold_method
    )

    rows = []
    for attack_id, payload in attack_to_nids.items():
        attack_nodes = payload["nids"]
        attack_indices = attack_windows.get(attack_id, [])
        for node_id in attack_nodes:
            if node_id not in node_to_scores:
                continue
            score = node_to_scores[node_id]
            if score <= thr:
                continue
            tw_idx = node_to_max_loss_tw.get(node_id, -1)
            stage = _stage_for_tw(tw_idx, attack_indices)
            rows.append(
                {
                    "node_id": int(node_id),
                    "attack_id": int(attack_id),
                    "tw_idx": int(tw_idx),
                    "stage": stage,
                    "score": float(score),
                    "threshold": float(thr),
                    "tw_file": filelist[tw_idx] if 0 <= tw_idx < len(filelist) else "",
                }
            )

    out_path = os.path.abspath(args.out)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
