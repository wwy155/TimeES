export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/TimeES.py \
    --dataset_type="Electricity" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:3" \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.8 \
    --pred_len=720 \
    runs  --seeds='[1,2,3,4,5]'


# python3 ./src/experiments/TimeES.py \
#     --dataset_type="ETTm1" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:3" \
#     --windows=96 \
#     --patience=5 \
#     --lr=0.0001 \
#     --M=96 \
#     --use_norm=False \
#     --tc_emb=False \
#     --t_emb=False \
#     --energy_ratio=0.8 \
#     --pred_len=720 \
#     config_wandb ForecastBase \
#     runs  --seeds='[1]'


python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:3" \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.6 \
    --pred_len=720 \
    config_wandb ForecastBase \
    runs  --seeds='[1]'


python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:3" \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=False \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.4 \
    --pred_len=720 \
    config_wandb ForecastBase \
    runs  --seeds='[1]'


# python3 ./src/experiments/TimeES.py \
#     --dataset_type="ETTm1" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:3" \
#     --windows=96 \
#     --patience=5 \
#     --lr=0.0001 \
#     --M=96 \
#     --use_norm=False \
#     --tc_emb=False \
#     --t_emb=False \
#     --energy_ratio=0.3 \
#     --pred_len=720 \
#     config_wandb ForecastBase \
#     runs  --seeds='[1,2,3,4,5]'

# python3 ./src/experiments/TimeES.py \
#     --dataset_type="ETTm1" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:3" \
#     --windows=96 \
#     --patience=5 \
#     --lr=0.0001 \
#     --M=96 \
#     --use_norm=False \
#     --tc_emb=False \
#     --t_emb=False \
#     --energy_ratio=0.1 \
#     --pred_len=720 \
#     config_wandb ForecastBase \
#     runs  --seeds='[1,2,3,4,5]'