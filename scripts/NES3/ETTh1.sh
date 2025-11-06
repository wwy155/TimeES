export PYTHONPATH=./:/notebooks/pytorchtimseries
python3 ./src/experiments/NES3.py \
    --dataset_type="ETTh1" \
    --batch_size=64 \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=64 \
    --columns="[5]" \
    --windows=96 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[133352]'
