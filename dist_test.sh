#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 CONFIG [GPUS] [EVAL_OPTIONS...]" >&2
    exit 2
fi

CONFIG=$1
GPUS=${2:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29503}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}" \
python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --nproc-per-node="$GPUS" \
    --master-port="$PORT" \
    "$SCRIPT_DIR/eval.py" \
    --config "$CONFIG" \
    --launcher pytorch \
    "${@:3}"
