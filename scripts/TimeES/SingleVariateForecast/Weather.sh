export PYTHONPATH=./
python3 ./src/experiments/TimeES.py \
    --dataset_type="Weather" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --tc_emb=False \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --layer_nums=2 \
    --energy_ratio=0.3 \
    --pred_len=96 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'



    export PYTHONPATH=./
python3 ./src/experiments/TimeES.py \
    --dataset_type="Weather" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --tc_emb=False \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --columns=[0] \
    --layer_nums=2 \
    --energy_ratio=0.3 \
    --pred_len=192 \
    config_wandb ForecastBaseBG \
    runs  --seeds='[1,2,3,4,5]'


export PYTHONPATH=./
python3 ./src/experiments/TimeES.py \
    --dataset_type="Weather" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --tc_emb=False \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --layer_nums=2 \
    --energy_ratio=0.3 \
    --pred_len=336 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'


export PYTHONPATH=./
python3 ./src/experiments/TimeES.py \
    --dataset_type="Weather" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --tc_emb=False \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --layer_nums=2 \
    --energy_ratio=0.3 \
    --pred_len=720 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'



