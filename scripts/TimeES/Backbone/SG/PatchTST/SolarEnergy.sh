export PYTHONPATH=./
# bash ./scripts/TimeES/SG/PatchTST/SolarEnergy.sh
python3 ./src/experiments/TimeES.py \
    --dataset_type="SolarEnergy" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:3" \
    --backbone='PatchTST' \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=True \
    --epochs=10 \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=1 \
    --pred_len=96 \
    --columns=[0] \
    config_wandb ForecastBaseSG \
    runs  --seeds='[1,2]'


python3 ./src/experiments/TimeES.py \
    --dataset_type="SolarEnergy" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:3" \
    --backbone='PatchTST' \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=True \
    --epochs=10 \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=1 \
    --pred_len=192 \
    --columns=[0] \
    config_wandb ForecastBaseSG \
    runs  --seeds='[1,2]'

python3 ./src/experiments/TimeES.py \
    --dataset_type="SolarEnergy" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:3" \
    --backbone='PatchTST' \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=True \
    --epochs=10 \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=1 \
    --pred_len=336 \
    --columns=[0] \
    config_wandb ForecastBaseSG \
    runs  --seeds='[1,2]'

python3 ./src/experiments/TimeES.py \
    --dataset_type="SolarEnergy" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:3" \
    --backbone='PatchTST' \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=True \
    --epochs=10 \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=1 \
    --pred_len=720 \
    --columns=[0] \
    config_wandb ForecastBaseSG \
    runs  --seeds='[1,2]'
