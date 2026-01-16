#!/bin/bash

export PYTHONPATH="./:/data/yww/notebook/pytorchtimseries"

DATA_PATH="/data/yww/notebook/pytorchtimseries/data"
DEVICE="cuda:0"
WINDOWS=336
REC_X=True
ENERGY_RATIO=0.9
PRED_LEN=336
SEEDS='[1,2,3,4,5]'

DATASET_TYPES=("$@")

if [ ${#DATASET_TYPES[@]} -eq 0 ]; then
    echo "Usage: $0 <dataset_type1> [dataset_type2] ... [dataset_typeN]"
    exit 1
fi

for dataset in "${DATASET_TYPES[@]}"; do
    echo "Running experiment for dataset: $dataset"
    python3 ./src/experiments/NESClassification.py \
        --dataset_type="$dataset" \
        --data_path="$DATA_PATH" \
        --device="$DEVICE" \
        --windows="$WINDOWS" \
        --rec_X="$REC_X" \
        --energy_ratio="$ENERGY_RATIO" \
        --pred_len="$PRED_LEN" \
        config_wandb ClassificationBase \
        runs --seeds="$SEEDS"
done
