export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESClassification.py \
    --dataset_type="StandWalkJump" \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --windows=2500 \
    --hidden_dim=512 \
    --use_norm=False \
    --lr=0.0001 \
    --patience=10 \
    --epochs=40 \
    --rec_X=True \
    --energy_ratio=0.9 \
    --pred_len=2500 \
    runs  --seeds='[1,2,3,4,5]'
