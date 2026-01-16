export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESProbForecast.py \
    --dataset_type="Electricity" \
    --batch_size=16 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=512 \
    --windows=168 \
    --patience=5 \
    --lr=0.001 \
    --M=168 \
    --use_norm=True \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=192 \
    config_wandb ProbForecastBase \
    runs  --seeds='[1,2,3,4,5]'


export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESProbForecast.py \
    --dataset_type="Electricity" \
    --batch_size=16 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=512 \
    --windows=168 \
    --patience=5 \
    --lr=0.001 \
    --M=168 \
    --use_norm=False \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=192 \
    config_wandb ProbForecastBase \
    runs  --seeds='[1,2,3,4,5]'