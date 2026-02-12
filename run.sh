#!/usr/bin/env bash

# Check if yq is installed
if ! command -v yq &> /dev/null; then
    echo "Error: yq is not installed. Please install it first:"
    echo "  - macOS: brew install yq"
    echo "  - Linux: wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && chmod +x /usr/bin/yq"
    exit 1
fi

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found in current directory"
    exit 1
fi

# Read parameters from config.yaml
DATA_ROOT=$(yq '.data_root' config.yaml)
NPROC=$(yq '.nproc_per_node' config.yaml)
EMBED_DIM=$(yq '.embed_dim' config.yaml)
DEPTH=$(yq '.depth' config.yaml)
NUM_HEADS=$(yq '.num_heads' config.yaml)
SOO_LR_SCALE=$(yq '.soo_lr_scale' config.yaml)
SAVE_FREQ=$(yq '.save_freq' config.yaml)
DROP_PATH=$(yq '.drop_path' config.yaml)
MODEL_EMA=$(yq '.model_ema' config.yaml)
USE_ORTHOGONAL=$(yq '.use_orthogonal' config.yaml)
USE_NO_ORTH_PROJECT_LAST=$(yq '.use_no_orth_project_last' config.yaml)
BATCH_SIZE=$(yq '.batch_size' config.yaml)
MIXUP_PROB=$(yq '.["mixup-prob"]' config.yaml)
MIXUP=$(yq '.mixup' config.yaml)
WEIGHT_DECAY=$(yq '.["weight-decay"]' config.yaml)

export MASTER_PORT=$((12000 + $RANDOM % 20000))

# Log all parameters
echo "========================================="
echo "Training Configuration (from config.yaml)"
echo "========================================="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Environment:"
echo "  MASTER_PORT: $MASTER_PORT"
echo ""
echo "Data:"
echo "  data_root: $DATA_ROOT"
echo ""
echo "Distributed Training:"
echo "  nproc_per_node: $NPROC"
echo ""
echo "Model Architecture:"
echo "  embed_dim: $EMBED_DIM"
echo "  depth: $DEPTH"
echo "  num_heads: $NUM_HEADS"
echo ""
echo "Training Parameters:"
echo "  batch_size: $BATCH_SIZE"
echo "  soo_lr_scale: $SOO_LR_SCALE"
echo "  save_freq: $SAVE_FREQ"
echo "  drop_path: $DROP_PATH"
echo "  model_ema: $MODEL_EMA"
echo "  mixup_prob: $MIXUP_PROB"
echo "  mixup: $MIXUP"
echo "  weight_decay: $WEIGHT_DECAY"
echo ""
echo "Orthogonal Settings:"
echo "  use_orthogonal: $USE_ORTHOGONAL"
echo "  use_no_orth_project_last: $USE_NO_ORTH_PROJECT_LAST"
echo "========================================="
echo ""

if [ "$USE_ORTHOGONAL" = true ]; then
    if [ "$USE_NO_ORTH_PROJECT_LAST" = true ]; then
        torchrun \
            --nproc_per_node $NPROC \
            --master_port $MASTER_PORT \
            train.py \
            --data $DATA_ROOT \
            --batch-size $BATCH_SIZE \
            --embed-dim $EMBED_DIM \
            --depth $DEPTH \
            --num-heads $NUM_HEADS \
            --model-ema \
            --soo-lr-scale $SOO_LR_SCALE \
            --save-freq $SAVE_FREQ \
            --drop-path $DROP_PATH \
            --mixup-prob $MIXUP_PROB \
            --mixup $MIXUP \
            --weight-decay $WEIGHT_DECAY \
            --orthogonal \
            --no-orth-project-last
    else
        torchrun \
            --nproc_per_node $NPROC \
            --master_port $MASTER_PORT \
            train.py \
            --data $DATA_ROOT \
            --batch-size $BATCH_SIZE \
            --embed-dim $EMBED_DIM \
            --depth $DEPTH \
            --num-heads $NUM_HEADS \
            --model-ema \
            --soo-lr-scale $SOO_LR_SCALE \
            --save-freq $SAVE_FREQ \
            --drop-path $DROP_PATH \
            --mixup-prob $MIXUP_PROB \
            --mixup $MIXUP \
            --weight-decay $WEIGHT_DECAY \
            --orthogonal
    fi
else
    if [ "$USE_NO_ORTH_PROJECT_LAST" = true ]; then
        torchrun \
            --nproc_per_node $NPROC \
            --master_port $MASTER_PORT \
            train.py \
            --data $DATA_ROOT \
            --batch-size $BATCH_SIZE \
            --embed-dim $EMBED_DIM \
            --depth $DEPTH \
            --num-heads $NUM_HEADS \
            --model-ema \
            --soo-lr-scale $SOO_LR_SCALE \
            --save-freq $SAVE_FREQ \
            --drop-path $DROP_PATH \
            --mixup-prob $MIXUP_PROB \
            --mixup $MIXUP \
            --weight-decay $WEIGHT_DECAY \
            --no-orth-project-last
    else
        torchrun \
            --nproc_per_node $NPROC \
            --master_port $MASTER_PORT \
            train.py \
            --data $DATA_ROOT \
            --batch-size $BATCH_SIZE \
            --embed-dim $EMBED_DIM \
            --depth $DEPTH \
            --num-heads $NUM_HEADS \
            --model-ema \
            --soo-lr-scale $SOO_LR_SCALE \
            --save-freq $SAVE_FREQ \
            --drop-path $DROP_PATH \
            --mixup-prob $MIXUP_PROB \
            --mixup $MIXUP \
            --weight-decay $WEIGHT_DECAY
    fi
fi
