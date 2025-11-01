export PYTHONPATH=./
python3 ./src/experiments/DLinear.py \
    --dataset_type="SimFreq3F" \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --windows=96 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[43,2,3,4,5]'
