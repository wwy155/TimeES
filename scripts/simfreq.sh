export PYTHONPATH=./
python3 ./src/experiments/ode_spec.py \
    --dataset_type="SimFreq5" \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --windows=336 \
    --fft_length=96 \
    --hop_length=96 \
    --recon_step=1 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[1631,2,3,4,5]'
