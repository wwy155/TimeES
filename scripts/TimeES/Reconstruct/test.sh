export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
# python3 ./src/experiments/TimeESReconstruct.py \
#     --dataset_type="ETTh1" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:2" \
#     --windows=168 \
#     --columns=[0] \
#     --patience=5 \
#     --lr=0.001 \
#     --M=96 \
#     --energy_ratio=0.9 \
#     --pred_len=1 \
#     runs  --seeds='[1]'


# python3 ./src/experiments/TimeESReconstruct.py \
#     --dataset_type="ETTh1" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:2" \
#     --windows=168 \
#     --columns=[0] \
#     --patience=5 \
#     --lr=0.001 \
#     --M=96 \
#     --energy_ratio=0.9 \
#     --pred_len=1 \
#     runs  --seeds='[1]'


# python3 ./src/experiments/TimeESReconstruct.py \
#     --dataset_type="ETTh2" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:2" \
#     --windows=168 \
#     --columns=[0] \
#     --patience=5 \
#     --lr=0.001 \
#     --M=96 \
#     --energy_ratio=0.9 \
#     --pred_len=1 \
#     runs  --seeds='[1]'


python3 ./src/experiments/TimeESReconstruct.py \
    --dataset_type="ETTm1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --windows=168 \
    --columns=[0] \
    --patience=5 \
    --lr=0.001 \
    --M=96 \
    --energy_ratio=1 \
    --pred_len=1 \
    runs  --seeds='[1]'


python3 ./src/experiments/TimeESReconstruct.py \
    --dataset_type="ETTm2" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --windows=168 \
    --columns=[0] \
    --patience=5 \
    --lr=0.001 \
    --M=96 \
    --energy_ratio=1 \
    --pred_len=1 \
    runs  --seeds='[1]'


# python3 ./src/experiments/TimeESReconstruct.py \
#     --dataset_type="Traffic" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:2" \
#     --windows=168 \
#     --columns=[0] \
#     --patience=5 \
#     --lr=0.001 \
#     --M=96 \
#     --energy_ratio=0.9 \
#     --pred_len=1 \
#     runs  --seeds='[1]'


# python3 ./src/experiments/TimeESReconstruct.py \
#     --dataset_type="SolarEnergy" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:2" \
#     --windows=168 \
#     --columns=[0] \
#     --patience=5 \
#     --lr=0.001 \
#     --M=96 \
#     --energy_ratio=0.9 \
#     --pred_len=1 \
#     runs  --seeds='[1]'

# python3 ./src/experiments/TimeESReconstruct.py \
#     --dataset_type="Electricity" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:2" \
#     --windows=168 \
#     --columns=[0] \
#     --patience=5 \
#     --lr=0.001 \
#     --M=96 \
#     --energy_ratio=0.9 \
#     --pred_len=1 \
#     runs  --seeds='[1]'

# python3 ./src/experiments/TimeESReconstruct.py \
#     --dataset_type="ILI" \
#     --batch_size=32 \
#     --data_path="/data/yww/notebook/pytorchtimseries/data" \
#     --device="cuda:2" \
#     --windows=168 \
#     --columns=[0] \
#     --patience=5 \
#     --lr=0.001 \
#     --M=96 \
#     --energy_ratio=0.9 \
#     --pred_len=1 \
#     runs  --seeds='[1]'