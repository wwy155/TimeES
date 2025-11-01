export PYTHONPATH=./
python3 ./src/experiments/ode_spec8.py \
    --dataset_type="ETTh1" \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --windows=336 \
    --train_time_domain=True \
    --fft_length=168 \
    --hop_length=168 \
    --c_emb_dim=64 \
    --recon_step=1 \
    --step_size=0.2 \
    --hidden_dim=512 \
    --lr=0.001 \
    --patience=5 \
    --pred_len=720 \
    runs  --seeds='[338331,2,3,4,5]'
