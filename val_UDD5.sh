python3 train.py \
    --cuda True \
    --resume checkpoints/ENet_UDD5_epoch300.pth.tar \
    --dataset UDD5 \
    --data_path ~/datasets/UDD5 \
    --model ENet \
    --init_eval False \
    --mode val