export PYTHONPATH=./
python3 ./src/experiments/ode_spec4.py \
    --dataset_type="ExchangeRate" \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --windows=96 \
    --fft_length=6 \
    --hop_length=6 \
    --recon_step=1 \
    --step_size=0.01 \
    --hidden_dim=512 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[163331,2,3,4,5]'
