#!/usr/bin/env bash
# Launch draft-head training on every visible GPU of this machine (data parallel).
# Usage: scripts/launch_training.sh [--model-config PATH] [--train-config PATH]
#        NUM_GPUS=4 scripts/launch_training.sh      # use a subset of GPUs
set -euo pipefail

cd "$(dirname "$0")/.."

NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "No CUDA GPU found; use 'python main.py train' for single-device training." >&2
    exit 1
fi

# Each process loads its own frozen target LLM and trains a replica of the head.
exec torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m training.train "$@"
