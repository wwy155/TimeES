export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES.py \
    --dataset_type="Traffic" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --additive_scale=False \
    --tc_emb=False \
    --t_emb=True \
    --energy_ratio=0.6 \
    --pred_len=96 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'



python3 ./src/experiments/NES.py \
    --dataset_type="Traffic" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --additive_scale=False \
    --tc_emb=False \
    --t_emb=True \
    --energy_ratio=0.6 \
    --pred_len=192 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'


python3 ./src/experiments/NES.py \
    --dataset_type="Traffic" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --additive_scale=False \
    --tc_emb=False \
    --t_emb=True \
    --energy_ratio=0.6 \
    --pred_len=336 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'


python3 ./src/experiments/NES.py \
    --dataset_type="Traffic" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --additive_scale=False \
    --tc_emb=False \
    --t_emb=True \
    --energy_ratio=0.6 \
    --pred_len=720 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'


