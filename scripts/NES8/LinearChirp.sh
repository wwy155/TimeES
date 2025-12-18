export PYTHONPATH=./:/notebooks/pytorchtimseries
python3 ./src/experiments/NES6.py \
    --dataset_type="LinearChirp" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=256 \
    --columns="[0]" \
    --windows=100 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=100 \
    runs  --seeds='[3143333]'
