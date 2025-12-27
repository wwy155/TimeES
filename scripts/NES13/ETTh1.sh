export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES11.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --hidden_dim=128 \
    --columns="[0]" \
    --windows=96 \
    --patience=5 \
    --topk=4 \
    --lr=0.001 \
    --M=512 \
    --pred_len=96 \
    runs  --seeds='[6617735]'
