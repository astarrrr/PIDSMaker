#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SYSTEM DATASET [extra args...]" >&2
  exit 1
fi

system="$1"
dataset="$2"
shift 2

seeds=(1 42 2020 2023)

for seed in "${seeds[@]}"; do
  echo "==> Running ${system} ${dataset} with seed=${seed}"
  PYTHONHASHSEED=0 python pidsmaker/main.py \
    "${system}" \
    "${dataset}" \
    --detection.gnn_training.use_seed=True \
    --detection.gnn_training.seed="${seed}" \
    --exp "${dataset}_${system}_seed${seed}" \
    --wandb \
    "$@"
done
