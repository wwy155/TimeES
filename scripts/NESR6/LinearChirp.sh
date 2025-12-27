export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NESR6.py \
    --dataset_type="LinearChirp" \
    --batch_size=300 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --columns="[0]" \
    --epochs=10000 \
    --lr=0.001 \
    --M=155  \
    --hop_length=1  \
    --K=155 \
    runs  --seeds='[77312223]'
