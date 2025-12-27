export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES13.py \
    --dataset_type="LinearChirp1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --columns="[0]" \
    --windows=200 \
    --lr=0.001 \
    --topk=1 \
    --patience=100 \
    --pred_len=200 \
    runs  --seeds='[634523]'
