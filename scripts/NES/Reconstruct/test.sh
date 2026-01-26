export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESReconstruct.py \
    --dataset_type="SimFreqCF" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --windows=500 \
    --columns=[0] \
    --patience=5 \
    --lr=0.001 \
    --additive_scale=False \
    --M=96 \
    --energy_ratio=1 \
    --pred_len=1 \
    runs  --seeds='[67876,2,3,4,5]'

