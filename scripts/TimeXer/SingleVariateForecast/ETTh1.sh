export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries

# Single-variate ETTh1 (select one channel via --columns)
python3 ./src/experiments/TimeXer.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --columns=[0] \
    --seq_len=96 \
    --patience=5 \
    --learning_rate=0.0001 \
    --train_epochs=100 \
    --task_name="long_term_forecast" \
    --features="S" \
    --use_norm=True \
    --patch_len=16 \
    --d_model=128 \
    --n_heads=8 \
    --e_layers=2 \
    --d_ff=512 \
    --dropout=0.1 \
    --embed="timeF" \
    --freq="h" \
    --pred_len=96 \
    runs --seeds='[1,2,3,4,5]'

python3 ./src/experiments/TimeXer.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --columns=[0] \
    --seq_len=96 \
    --patience=5 \
    --learning_rate=0.0001 \
    --train_epochs=100 \
    --task_name="long_term_forecast" \
    --features="S" \
    --use_norm=True \
    --patch_len=16 \
    --d_model=128 \
    --n_heads=8 \
    --e_layers=2 \
    --d_ff=512 \
    --dropout=0.1 \
    --embed="timeF" \
    --freq="h" \
    --pred_len=192 \
    runs --seeds='[1,2,3,4,5]'

python3 ./src/experiments/TimeXer.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --columns=[0] \
    --seq_len=96 \
    --patience=5 \
    --learning_rate=0.0001 \
    --train_epochs=100 \
    --task_name="long_term_forecast" \
    --features="S" \
    --use_norm=True \
    --patch_len=16 \
    --d_model=128 \
    --n_heads=8 \
    --e_layers=2 \
    --d_ff=512 \
    --dropout=0.1 \
    --embed="timeF" \
    --freq="h" \
    --pred_len=336 \
    runs --seeds='[1,2,3,4,5]'

python3 ./src/experiments/TimeXer.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --columns=[0] \
    --seq_len=96 \
    --patience=5 \
    --learning_rate=0.0001 \
    --train_epochs=100 \
    --task_name="long_term_forecast" \
    --features="S" \
    --use_norm=True \
    --patch_len=16 \
    --d_model=128 \
    --n_heads=8 \
    --e_layers=2 \
    --d_ff=512 \
    --dropout=0.1 \
    --embed="timeF" \
    --freq="h" \
    --pred_len=720 \
    runs --seeds='[1,2,3,4,5]'

