#!/bin/bash

# This script forwards all command-line arguments to the Python script

# Check if any arguments were provided
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

# Resolve repo root based on this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Construct the argument string to pass to the Python script
args=""
for arg in "$@"; do
    args+="$arg "
done

# Optional restart flag (set RESTART_FROM_SCRATCH=1 to force a fresh run)
if [ "${RESTART_FROM_SCRATCH}" = "1" ]; then
    args+=" --restart_from_scratch "
fi

# Use local NLTK data if available
if [ -z "${NLTK_DATA}" ] && [ -d "${REPO_DIR}/nltk_data" ]; then
    export NLTK_DATA="${REPO_DIR}/nltk_data"
fi

# Execute the Python script with the passed arguments
PYTHONHASHSEED=0 nohup python "${REPO_DIR}/pidsmaker/main.py" $args --wandb &



