#!/bin/bash

model=${1:-NES5}
dataset_type=${2:-ETTh1}
device=${3:-cuda:0}
windows=${4:-96}
pred_len=${5:-96}

export PYTHONPATH=./

python3 ./src/experiments/$model.py \
    --dataset_type "$dataset_type" \
    --columns "[0]" \
    --windows $windows \
    --hidden_dim=512 \
    --pred_len $pred_len \
    --lr 0.0001 \
    --additive_scale True \
    --device "$device" \
    config_wandb 3902NES \
    runs --seeds='[1,2,3,4,5]'





