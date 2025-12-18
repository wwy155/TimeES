export PYTHONPATH=./:/notebooks/pytorchtimseries
python3 ./src/experiments/DLinear.py \
    --dataset_type="Traffic" \
    --data_path="/notebooks/pytorchtimseries/data" \
    --columns="[0]" \
    --device="cuda:0" \
    --windows=96 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[132456,2,3,4,5]'
