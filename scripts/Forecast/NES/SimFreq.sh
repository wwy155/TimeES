export PYTHONPATH=./:/notebooks/pytorchtimseries
python3 ./src/experiments/NESForecast2.py \
    --dataset_type="SimFreq" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --columns="[0]" \
    --windows=700 \
    --lr=0.001 \
    --patience=5 \
    --M=300 \
    --energy_ratio=0.9 \
    --pred_len=700 \
    runs  --seeds='[31465633]'
