# export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
# python3 ./src/experiments/TimeES.py \
#     --dataset_type="ETTh2" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:2" \
#     --backbone='linear' \
#     --windows=96 \
#     --patience=5 \
#     --lr=0.0001 \
#     --M=96 \
#     --use_norm=True \
#     --epochs=10 \
#     --tc_emb=False \
#     --t_emb=False \
#     --energy_ratio=0.001 \
#     --pred_len=720 \
#     runs  --seeds='[54445334,2,3]'

export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/TimeESProbForecast.py \
    --dataset_type="SolarEnergy" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:1" \
    --hidden_dim=512 \
    --windows=168 \
    --patience=5 \
    --lr=0.001 \
    --M=168 \
    --use_norm=True \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.001 \
    --pred_len=192 \
    runs  --seeds='[13,2,3,4,5]'