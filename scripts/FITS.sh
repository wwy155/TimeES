export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/FITS.py \
    --dataset_type="LinearChirp1" \
    --columns="[0]" \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --windows=96 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[661235,2,3,4,5]'
