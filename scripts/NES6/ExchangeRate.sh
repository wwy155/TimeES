export PYTHONPATH=./:/notebooks/pytorchtimseries
python3 ./src/experiments/NES4.py \
    --dataset_type="ExchangeRate" \
    --batch_size=32 \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=256 \
    --columns="[5]" \
    --windows=336 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[15353]'
