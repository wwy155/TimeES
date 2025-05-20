export PYTHONPATH=./
python3 ./src/experiments/ode_spec5.py \
    --dataset_type="ETTh1" \
    --data_path="/notebooks/pytorchtimseries/data" \
    --device="cuda:0" \
    --windows=96 \
    --fft_length=24 \
    --hop_length=24 \
    --recon_step=1 \
    --step_size=0. \
    --hidden_dim=512 \
    --patience=5 \
    --pred_len=720 \
    runs  --seeds='[331,2,3,4,5]'
