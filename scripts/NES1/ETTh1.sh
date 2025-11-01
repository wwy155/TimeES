export PYTHONPATH=./:/notebooks/pytorchtimseries
python3 ./src/experiments/NES1.py \
    --dataset_type="ETTh1" \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=128 \
    --columns="[6]" \
    --nw=10 \
    --windows=10 \
    --lr=0.01 \
    --patience=5 \
    --pred_len=10 \
    runs  --seeds='[12324345,2,3,4,5]'
