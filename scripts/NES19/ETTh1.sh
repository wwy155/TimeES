export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES19.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=512 \
    --columns="[0]" \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --energy_ratio=0.9 \
    --pred_len=96 \
    runs  --seeds='[434343]'
