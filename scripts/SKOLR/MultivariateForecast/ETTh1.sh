#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-./:/data/yww/notebook/pytorchtimseries}"

python3 src/experiments/SKOLR.py \
  --dataset_type ETTh1 \
  --data_path /data/yww/notebook/pytorchtimseries/data \
  --device cuda:0 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --dynamic_dim 128 \
  --hidden_dim 64 \
  --hidden_layers 2 \
  --seg_len 48 \
  --num_blocks 3 \
  --dropout 0.05 \
  --mask_type global \
  config_wandb ForecastBase \
  runs --seeds='[1,2,3,4,5]'

