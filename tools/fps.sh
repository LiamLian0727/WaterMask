#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
GPUS=${GPUS:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/analysis_tools/benchmark.py \
    $CONFIG \
    $CHECKPOINT \
    --log-interval 50 \
    --max-iter 691 \
    --launcher pytorch ${@:3}