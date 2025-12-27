export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES9.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=256 \
    --columns="[0]" \
    --windows=200 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=200 \
    runs  --seeds='[991111]'
