export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES.py \
    --dataset_type="SimFreqCF" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --windows=96 \
    --patience=5 \
    --lr=0.001 \
    --M=512 \
    --use_norm=False \
    --tc_emb=False \
    --t_emb=False \
    --epochs=100 \
    --energy_ratio=1 \
    --pred_len=96 \
    config_wandb ForecastBase \
    runs  --seeds='[1,2,3,4,5]'

