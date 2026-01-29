export PYTHONPATH=./
python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --use_norm=True \
    --M=96 \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=720 \
    runs  --seeds='[1]'
