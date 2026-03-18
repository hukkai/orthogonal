#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

ORTH_TYPE=$1
ORTH_DIM=$2
# DATA_ROOT="/project/flame/kaihu/imagenet"
DATA_ROOT="/opt/dlami/nvme/imagenet"

ORTH_ARGS=()
if [ -n "$ORTH_DIM" ]; then
    ORTH_ARGS+=(--orth-dim "$ORTH_DIM")
fi

torchrun \
    --nproc_per_node 8 \
    --master_port $MASTER_PORT \
    train.py \
    --data $DATA_ROOT \
    --embed-dim 1024 \
    --depth 24 \
    --num-heads 16 \
    --model-ema \
    --orthogonal-type $ORTH_TYPE \
    "${ORTH_ARGS[@]}"
