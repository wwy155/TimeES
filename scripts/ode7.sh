export PYTHONPATH=./
python3 ./src/experiments/ode_spec7.py \
    --dataset_type="ExchangeRate" \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --windows=96 \
    --train_time_domain=True \
    --fft_length=24 \
    --hop_length=24 \
    --recon_step=1 \
    --step_size=0.2 \
    --hidden_dim=512 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=96 \
    runs  --seeds='[33831,2,3,4,5]'
