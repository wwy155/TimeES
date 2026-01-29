export PYTHONPATH=./
python3 ./src/experiments/TimeES.py \
    --dataset_type="Traffic" \
    --batch_size=1 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --hidden_dim=512 \
    --windows=168 \
    --patience=5 \
    --lr=0.001 \
    --M=168 \
    --use_norm=True \
    --tc_emb=False \
    --t_emb=True \
    --energy_ratio=0.5 \
    --pred_len=192 \
    config_wandb ProbForecastBase \
    runs  --seeds='[1,2,3,4,5]'


export PYTHONPATH=./
python3 ./src/experiments/TimeES.py \
    --dataset_type="Traffic" \
    --batch_size=1 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --hidden_dim=512 \
    --windows=168 \
    --patience=5 \
    --lr=0.001 \
    --M=168 \
    --use_norm=False \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.5 \
    --pred_len=192 \
    config_wandb ProbForecastBase \
    runs  --seeds='[1,2,3,4,5]'