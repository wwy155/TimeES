export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESR5.py \
    --dataset_type="ETTh1" \
    --batch_size=300 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --columns="[0]" \
    --epochs=10000 \
    --lr=0.001 \
    --M=200  \
    --K=100 \
    runs  --seeds='[623]'
