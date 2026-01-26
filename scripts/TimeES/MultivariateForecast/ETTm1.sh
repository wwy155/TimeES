export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --device="cuda:1" \
    --use_norm=True \
    --t_emb=True \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --energy_ratio=0.9 \
    --pred_len=96 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'



python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --device="cuda:1" \
    --use_norm=True \
    --t_emb=True \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --energy_ratio=0.9 \
    --pred_len=192 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'

python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --device="cuda:1" \
    --use_norm=True \
    --t_emb=True \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --energy_ratio=0.9 \
    --pred_len=336 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'

python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --device="cuda:1" \
    --use_norm=True \
    --t_emb=True \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --energy_ratio=0.9 \
    --pred_len=720 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'

python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --device="cuda:1" \
    --use_norm=True \
    --t_emb=True \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --energy_ratio=0.9 \
    --pred_len=720 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'