#!/bin/bash

dataset_type=${1:-ETTh1}
device=${2:-cuda:0}
windows=${3:-96}
pred_len=${4:-96}

export PYTHONPATH=./

python3 ./src/experiments/NES20.py \
    --dataset_type "$dataset_type" \
    --windows $windows \
    --hidden_dim=512 \
    --pred_len $pred_len \
    --lr 0.0001 \
    --energy_ratio=0.9 \
    --additive_scale True \
    --device "$device" \
    config_wandb 3902NES \
    runs --seeds='[1,2,3,4,5]'





