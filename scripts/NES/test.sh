export PYTHONPATH=./:/data/yww/notebook/pytorchtimseries
python3 ./src/experiments/NES.py \
    --dataset_type="ETTh1" \
    --batch_size=32 \
    --data_path="/data/yww/notebook/pytorchtimseries/data" \
    --device="cuda:2" \
    --hidden_dim=512 \
    --windows=96 \
    --columns=[0] \
    --patience=5 \
    --lr=0.0001 \
    --M=96 \
    --use_norm=True \
    --tc_emb=False \
    --t_emb=False \
    --energy_ratio=0.9 \
    --pred_len=720 \
    runs  --seeds='[1,2,3,4,5]'




    # bash ./scripts/NES/run_mv_wandb.sh cuda:1 Electricity  SolarEnergy
    # bash ./scripts/NES/run_mv_wandb.sh cuda:3  Traffic 
    # bash ./scripts/NES/Weather.sh