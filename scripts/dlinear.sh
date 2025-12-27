export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/DLinear.py \
    --dataset_type="ETTh1" \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --columns="[0]" \
    --device="cuda:0" \
    --windows=96 \
    --lr=0.0001 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[132456,2,3,4,5]'
