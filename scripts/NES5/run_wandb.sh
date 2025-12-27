export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES5.py \
    --dataset_type="LinearChirp1" \
    --batch_size=512 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=256 \
    --columns="[0]" \
    --windows=96 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[1,2,3,4,5]'
