export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES17.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=512 \
    --columns="[0]" \
    --windows=96 \
    --lr=0.0001 \
    --patience=5 \
    --pred_len=96 \
    --topk=49 \
    runs  --seeds='[65776512]'
