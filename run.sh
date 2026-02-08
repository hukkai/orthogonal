#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

DATA_ROOT="/project/flame/kaihu/imagenet"
torchrun \
    --nproc_per_node 8 \
    --master_port $MASTER_PORT \
    train.py \
    --data $DATA_ROOT \
    --embed-dim 1024 \
    --depth 24 \
    --num-heads 16 \
    --model-ema \
    --orthogonal
