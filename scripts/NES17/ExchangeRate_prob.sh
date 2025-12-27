export PYTHONPATH=./:/notebooks/pytorchtimseries
python3 ./src/experiments/NESprob.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=256 \
    --columns="[4]" \
    --windows=96 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[15353]'
