export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES12.py \
    --dataset_type="LinearChirp" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=256 \
    --columns="[0]" \
    --windows=96 \
    --lr=0.001 \
    --M=9 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[434223]'
