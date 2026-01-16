
export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESForecast2.py \
    --dataset_type="Traffic" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --columns=[0] \
    --device="cuda:1" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --device="cuda:2" \
    --t_emb=True \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --energy_ratio=0.6 \
    --pred_len=336 \
    runs  --seeds='[1,2,3,4,5]'


export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESForecast2.py \
    --dataset_type="Traffic" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --columns=[0] \
    --device="cuda:1" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --device="cuda:2" \
    --t_emb=True \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --energy_ratio=0.6 \
    --pred_len=192 \
    runs  --seeds='[1,2,3,4,5]'



export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESForecast2.py \
    --dataset_type="Traffic" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --columns=[0] \
    --device="cuda:1" \
    --hidden_dim=512 \
    --additive_scale=False \
    --windows=96 \
    --device="cuda:2" \
    --t_emb=True \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --energy_ratio=0.6 \
    --pred_len=720 \
    runs  --seeds='[1,2,3,4,5]'
