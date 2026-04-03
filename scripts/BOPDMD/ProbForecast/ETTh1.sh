export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
# BOP-DMD is non-parametric: no SGD; learning_rate / train_epochs are ignored (eval-only run).
python3 ./src/experiments/BOPDMD.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:0" \
    --seq_len=169 \
    --patience=5 \
    --learning_rate=0.001 \
    --svd_rank=1 \
    --num_trials=50 \
    --trial_size=0.9 \
    --eig_constraints="stable" \
    --num_samples=20 \
    --pred_len=192 \
    runs --seeds='[2222,2,3,4,5]'

