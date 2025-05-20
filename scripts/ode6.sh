export PYTHONPATH=./
python3 ./src/experiments/ode_spec6.py \
    --dataset_type="ETTh1" \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --windows=336 \
    --fft_length=168 \
    --hop_length=168 \
    --train_time_domain=True \
    --recon_step=1 \
    --step_size=0.2 \
    --hidden_dim=128 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=336 \
    runs  --seeds='[33831,2,3,4,5]'
