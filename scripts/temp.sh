export PYTHONPATH=./
python3 ./src/experiments/ode_spec.py \
    --dataset_type="ExchangeRate" \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --windows=336 \
    --fft_length=96 \
    --hop_length=96 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[61121,2,3,4,5]'
