export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES11.py \
    --dataset_type="ExchangeRate" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=256 \
    --columns="[0]" \
    --windows=96 \
    --lr=0.001 \
    --M=96 \
    --patience=5 \
    --topk=1 \
    --pred_len=96 \
    runs  --seeds='[765653232]'
