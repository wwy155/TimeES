#!/bin/bash

dataset_type=${1:-ETTh1}
device=${2:-cuda:0}
windows=${3:-96}
pred_len=${4:-96}

export PYTHONPATH=./

python3 ./src/experiments/NES15.py \
    --dataset_type "$dataset_type" \
    --columns "[0]" \
    --windows $windows \
    --pred_len $pred_len \
    --lr 0.001 \
    --additive_scale True \
    --device "$device" \
    runs --seeds='[1,2,3,4,5]'





