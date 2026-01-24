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

# Resolve repo root based on this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Optional restart flag (set RESTART_FROM_SCRATCH=1 to force a fresh run)
extra_args=("$@")
if [[ "${RESTART_FROM_SCRATCH:-}" == "1" ]]; then
  extra_args+=("--restart_from_scratch")
fi

# Use local NLTK data if available
if [[ -z "${NLTK_DATA:-}" && -d "${REPO_DIR}/nltk_data" ]]; then
  export NLTK_DATA="${REPO_DIR}/nltk_data"
fi

log_dir="nohup_logs"
mkdir -p "${log_dir}"

for seed in "${seeds[@]}"; do
  log_file="${log_dir}/nohup_seed_${seed}.out"
  echo "==> Running ${system} ${dataset} with seed=${seed}, log=${log_file}"
  PYTHONHASHSEED=0 nohup python "${REPO_DIR}/pidsmaker/main.py" \
    "${system}" \
    "${dataset}" \
    --detection.gnn_training.use_seed=True \
    --detection.gnn_training.seed="${seed}" \
    --exp "${dataset}_${system}_seed${seed}" \
    "${extra_args[@]}" \
    --wandb \
    > "${log_file}" 2>&1 &
  pid=$!
  wait "${pid}"
done
