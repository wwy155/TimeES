export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESprob1.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=256 \
    --columns="[0]" \
    --windows=96 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[15353]'
