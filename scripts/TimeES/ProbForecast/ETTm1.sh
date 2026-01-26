export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
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
    --energy_ratio=1 \
    --pred_len=192 \
    config_wandb ProbForecastBase \
    runs  --seeds='[1,2,3,4,5]'
