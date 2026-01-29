export PYTHONPATH=./
python3 ./src/experiments/TimeES.py \
    --dataset_type="ExchangeRate" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --windows=168 \
    --patience=5 \
    --lr=0.001 \
    --M=168 \
    --use_norm=True \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=192 \
    runs  --seeds='[233332, 2,3,4,5]'