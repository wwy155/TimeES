export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESClassification.py \
    --dataset_type="PhonemeSpectra" \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:3" \
    --windows=336 \
    --hidden_dim=512 \
    --use_norm=False \
    --addictive_scale=False \
    --lr=0.0001 \
    --patience=10 \
    --energy_ratio=1 \
    --pred_len=336 \
    runs  --seeds='[424,2,3,4,5]'
