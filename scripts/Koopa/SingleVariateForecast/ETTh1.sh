#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-./:/data/yww/notebook/pytorchtimseries}"

python3 src/experiments/Koopa.py \
  --dataset_type ETTh1 \
  --data_path /data/yww/notebook/pytorchtimseries/data \
  --device cuda:0 \
  --columns [0] \
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
  --alpha 0.2 \
  config_wandb ForecastBaseSG \
  runs --seeds='[1,2,3,4,5]'

