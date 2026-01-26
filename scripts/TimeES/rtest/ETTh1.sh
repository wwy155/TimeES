export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/TimeESTestK.py \
    --dataset_type="SolarEnergy" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --backbone='linear' \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=True \
    --epochs=10 \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=36 \
    runs  --seeds='[1,2,3]'

