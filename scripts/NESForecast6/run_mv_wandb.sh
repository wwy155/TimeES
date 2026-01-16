#!/bin/bash

export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries

BATCH_SIZE=16
DATA_PATH="/data/yww/notebook/pytorchtimseries/data"
HIDDEN_DIM=128
WINDOWS=96
PATIENCE=5
LR=0.0001
M=96
ENERGY_RATIO=0.9
PRED_LEN=96
SEEDS='[1,2,3,4,5]'
PRED_LEN_LIST=(96 168 336 720)
T_EMB=False
ADDTIVE_SCALE=False

if [ $# -lt 2 ]; then
    echo "Usage: $0 <device> <dataset1> [dataset2] ... [datasetN]"
    echo "Example: $0 cuda:1 ETTh1 ETTh2 Weather"
    exit 1
fi

DEVICE="$1"
shift  
DATASETS=("$@")

echo "Device: $DEVICE"
echo "Datasets: ${DATASETS[*]}"
echo "----------------------------------------"
for dataset in "${DATASETS[@]}"; do
    echo ">>> Running on dataset: $dataset with pred_len=$PRED_LEN"
    for PRED_LEN in "${PRED_LEN_LIST[@]}"; do
        python3 ./src/experiments/NESForecast4.py \
            --dataset_type="$dataset" \
            --batch_size="$BATCH_SIZE" \
            --data_path="$DATA_PATH" \
            --device="$DEVICE" \
            --additive_scale=$ADDTIVE_SCALE \
            --t_emb=$T_EMB \
            --hidden_dim="$HIDDEN_DIM" \
            --windows="$WINDOWS" \
            --patience="$PATIENCE" \
            --lr="$LR" \
            --M="$M" \
            --energy_ratio="$ENERGY_RATIO" \
            --pred_len="$PRED_LEN" \
            config_wandb ForecastBase \
            runs --seeds="$SEEDS"
    done
done


echo "All done!"
