#!/bin/bash

# 默认参数
device="cuda:0"
runs="[1,2,3,4,5]"
task="UEAClassification"
# bash ./scripts/UEAClassification/run_wandb.sh
# 可选：从命令行传入 device
if [ $# -ge 1 ]; then
    device="$1"
fi
# 模型列表（可自定义）
model_list=("FreTS" "Informer")

# 数据集列表（可包含重复，但通常不必要）
dataset_type_list=(
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
    "SelfRegulationSCP1"
    "SelfRegulationSCP2"
    "StandWalkJump"
    "UWaveGestureLibrary"
    "Handwriting"
    "Heartbeat"
    "SpokenArabicDigits"
)

# 遍历所有组合
for model in "${model_list[@]}"; do
    for dataset in "${dataset_type_list[@]}"; do
        echo "Running: model=$model, dataset_type=$dataset, device=$device"

        # 根据数据集设置 series_length（来自你提供的官方值）
        case "$dataset" in
            "EthanolConcentration")   series_length=1751 ;;
            "FaceDetection")          series_length=62   ;;
            "JapaneseVowels")         series_length=29   ;;
            "Libras")                 series_length=45   ;;
            "MotorImagery")           series_length=1001 ;;
            "NATOPS")                 series_length=51   ;;
            "PEMS-SF")                series_length=144  ;;
            "PenDigits")              series_length=8    ;;
            "PhonemeSpectra")         series_length=217  ;;
            "RacketSports")           series_length=30   ;;
            "SelfRegulationSCP1")     series_length=896  ;;
            "SelfRegulationSCP2")     series_length=1152 ;;
            "StandWalkJump")          series_length=2500 ;;
            "UWaveGestureLibrary")    series_length=315  ;;
            "Handwriting")            series_length=152  ;;
            "Heartbeat")              series_length=231  ;;
            "SpokenArabicDigits")     series_length=93  ;;
            *)
                echo "Error: Unknown dataset '$dataset'. Please add its series_length." >&2
                exit 1
                ;;
        esac
# pytexp --model FreTS --task UEAClassification --dataset_type StandWalkJump --device cuda:0 runs "[1,2,3,4,5]

        # 构建并执行命令
        pytexp \
            --model "$model" \
            --task "$task" \
            --dataset_type "$dataset" \
            --device="$device" \
            config_wandb ClassificationBase \
            runs "$runs"
    done
done

echo "All experiments completed."
