#!/bin/bash

export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries

# 默认参数（可按需修改）
BATCH_SIZE=32
DATA_PATH="/data/yww/notebook/pytorchtimseries/data"
HIDDEN_DIM=512
WINDOWS=96
PATIENCE=5
LR=0.0001
M=96
ENERGY_RATIO=0.9
PRED_LEN=96
SEEDS='[1,2,3,4,5]'

# 解析命令行参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 <device> <dataset1> [dataset2] ... [datasetN]"
    echo "Example: $0 cuda:1 ETTh1 ETTh2 Weather"
    exit 1
fi

DEVICE="$1"
shift  # 移除第一个参数（device），剩下的都是数据集
DATASETS=("$@")

echo "Device: $DEVICE"
echo "Datasets: ${DATASETS[*]}"
echo "----------------------------------------"

# 遍历每个数据集并运行
for dataset in "${DATASETS[@]}"; do
    echo ">>> Running on dataset: $dataset"
    python3 ./src/experiments/NESPatchTST.py \
        --dataset_type="$dataset" \
        --batch_size="$BATCH_SIZE" \
        --data_path="$DATA_PATH" \
        --device="$DEVICE" \
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

echo "All done!"
