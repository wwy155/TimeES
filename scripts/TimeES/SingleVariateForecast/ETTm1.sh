export PYTHONPATH=./
python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --columns=[0] \
    --hidden_dim=512 \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=96 \
    config_wandb ForecastBaseSG \
    runs  --seeds='[1,2,3,4,5]'



python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --columns=[0] \
    --hidden_dim=512 \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=192 \
    config_wandb ForecastBaseSG \
    runs  --seeds='[1,2,3,4,5]'



python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --columns=[0] \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --hidden_dim=512 \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=336 \
    config_wandb ForecastBaseSG \
    runs  --seeds='[1,2,3,4,5]'


python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --columns=[0] \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --hidden_dim=512 \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=720 \
    config_wandb ForecastBaseSG \
    runs  --seeds='[1,2,3,4,5]'
