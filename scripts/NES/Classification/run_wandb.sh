#!/bin/bash
export PYTHONPATH=./

device="cuda:0"
runs="[1,2,3,4,5]"
task="UEAClassification"

# bash ./scripts/NES/Classification/run_wandb.sh cuda:2 1 False "EthanolConcentration MotorImagery Handwriting Heartbeat SpokenArabicDigits"
# bash ./scripts/NES/Classification/run_wandb.sh cuda:3 0.9 False "EthanolConcentration MotorImagery Handwriting Heartbeat SpokenArabicDigits"
# bash ./scripts/NES/Classification/run_wandb.sh cuda:1 1 True "EthanolConcentration MotorImagery Handwriting Heartbeat SpokenArabicDigits"
# bash ./scripts/NES/Classification/run_wandb.sh cuda:3 0.9 True   "EthanolConcentration MotorImagery Handwriting Heartbeat SpokenArabicDigits"

ENERGY_RATIO=1
USE_NORM=False

if [ $# -ge 1 ]; then
    device="$1"
fi
if [ $# -ge 2 ]; then
    ENERGY_RATIO="$2"
fi

if [ $# -ge 3 ]; then
    USE_NORM="$3"
fi

# )
# if [ $# -ge 2 ]; then
#     read -ra model_list <<< "$2"
# else
#     model_list=("${default_models[@]}")
# fi

dataset_type_list=(
    "StandWalkJump"
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
    "Handwriting"
    "Heartbeat"
    "SpokenArabicDigits"
)

if [ $# -ge 4 ]; then
    read -ra dataset_type_list <<< "$4"
else
    dataset_type_list=("${dataset_type_list[@]}")
fi


# 遍历所有组合
for dataset in "${dataset_type_list[@]}"; do
    echo "Running, dataset_type=$dataset, device=$device"

    # 根据数据集设置 series_length（来自你提供的官方值）
    case "$dataset" in
            # "EthanolConcentration")   series_length=1751 ;;
            "EthanolConcentration")   series_length=336 ;;
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
            "Handwriting")    series_length=152  ;;
            "Heartbeat")    series_length=231  ;;
            "SpokenArabicDigits")    series_length=93  ;;
        *)
            echo "Error: Unknown dataset '$dataset'. Please add its series_length." >&2
            exit 1
            ;;
    esac

    # 构建并执行命令
    python3 ./src/experiments/NESClassification.py \
        --model "$model" \
        --batch_size=8 \
        --energy_ratio="$ENERGY_RATIO" \
        --dataset_type "$dataset" \
        --use_norm=$USE_NORM \
        --device="$device" \
        --windows "$series_length" \
        config_wandb ClassificationBase \
        runs "$runs"
done

echo "All experiments completed."
