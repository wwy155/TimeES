export PYTHONPATH=./:/notebooks/pytorchtimseries
python3 ./src/experiments/NES2.py \
    --dataset_type="ExchangeRate" \
    --batch_size=24 \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=512 \
    --columns="[5]" \
    --max_w=0.5 \
    --nw=1024 \
    --windows=10 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=10 \
    runs  --seeds='[13333352]'
