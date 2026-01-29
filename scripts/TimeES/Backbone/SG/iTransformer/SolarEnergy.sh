export PYTHONPATH=./
# bash ./scripts/TimeES/Backbone/iTransformer/SolarEnergy.sh
# bash ./scripts/TimeES/Backbone/PatchTST/SolarEnergy.sh
# bash ./scripts/TimeES/Backbone/PatchTST/ExchangeRate.sh
# bash ./scripts/TimeES/Backbone/iTransformer.SolarEnergy
python3 ./src/experiments/TimeES.py \
    --dataset_type="SolarEnergy" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --backbone='iTransformer' \
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
    runs  --seeds='[1,2,3]'


python3 ./src/experiments/TimeES.py \
    --dataset_type="SolarEnergy" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --columns=[0] \
    --device="cuda:1" \
    --backbone='iTransformer' \
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
    runs  --seeds='[1,2,3]'

python3 ./src/experiments/TimeES.py \
    --dataset_type="SolarEnergy" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --columns=[0] \
    --device="cuda:1" \
    --backbone='iTransformer' \
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
    runs  --seeds='[1,2,3]'


python3 ./src/experiments/TimeES.py \
    --dataset_type="SolarEnergy" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --backbone='iTransformer' \
    --windows=96 \
    --columns=[0] \
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
    runs  --seeds='[1,2,3]'

