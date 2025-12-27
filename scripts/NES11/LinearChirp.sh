export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES11.py \
    --dataset_type="LinearChirp" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --columns="[0]" \
    --windows=96 \
    --lr=0.001 \
    --topk=22 \
    --M=512 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[938398]'
