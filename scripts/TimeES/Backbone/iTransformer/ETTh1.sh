export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
# bash ./scripts/TimeES/Backbone/iTransformer/ETTh1.sh
# bash ./scripts/TimeES/Backbone/PatchTST/ETTh1.sh
# bash ./scripts/TimeES/Backbone/PatchTST/ExchangeRate.sh
# bash ./scripts/TimeES/Backbone/iTransformer.ETTh1
python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --backbone='iTransformer' \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=True \
    --epochs=10 \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=96 \
    runs  --seeds='[1,2,3]'


python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --backbone='iTransformer' \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=True \
    --epochs=10 \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=192 \
    runs  --seeds='[1,2,3]'

python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --backbone='iTransformer' \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=True \
    --epochs=10 \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=336 \
    runs  --seeds='[1,2,3]'

python3 ./src/experiments/TimeES.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --backbone='iTransformer' \
    --windows=96 \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=True \
    --epochs=10 \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=720 \
    runs  --seeds='[1,2,3]'
