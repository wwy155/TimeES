export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES8.py \
    --dataset_type="ExchangeRate" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=128 \
    --columns="[0]" \
    --windows=96 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[132456]'
