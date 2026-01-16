#!/bin/bash

# 固定参数
task="UEAClassification"
runs="[1,2,3,4,5]"

# 默认 device
device="cuda:0"

# 默认模型列表（可被命令行覆盖）
default_models=("DLinear")
# 默认数据集列表：全部 14 个 UEA 数据集
default_datasets=(
    "SelfRegulationSCP1"
    "SelfRegulationSCP2"
    "EthanolConcentration"
    "FaceDetection"
    "JapaneseVowels"
    "Libras"
    "MotorImagery"
    "NATOPS"
    "PEMS-SF"
    "PenDigits"
    "PhonemeSpectra"
    "RacketSports"
    "StandWalkJump"
    "UWaveGestureLibrary"
)

# 解析命令行参数
if [ $# -ge 1 ]; then
    device="$1"
fi
if [ $# -ge 2 ]; then
    read -ra model_list <<< "$2"
else
    model_list=("${default_models[@]}")
fi
if [ $# -ge 3 ]; then
    read -ra dataset_type_list <<< "$3"
else
    dataset_type_list=("${default_datasets[@]}")
fi

echo "Device: $device"
echo "Models: ${model_list[*]}"
echo "Datasets: ${dataset_type_list[*]}"
echo "----------------------------------------"

# 遍历所有组合
for model in "${model_list[@]}"; do
    for dataset in "${dataset_type_list[@]}"; do
        echo "Running: model=$model, dataset=$dataset, device=$device"
        python3 ./src/experiments/NESClassification.py \
            --model "$model" \
            --task "$task" \
            --dataset_type "$dataset" \
            --windows "336" \
            --device="$device" \
            config_wandb ClassificationBase \
            runs "$runs"
    done
done
