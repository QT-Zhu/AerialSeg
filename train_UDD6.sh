python3 train.py \
    --cuda True \
    --epoch 300 \
    --train_batch_size 8 \
    --dataset UDD6 \
    --data_path ~/datasets/UDD6 \
    --model GCN \
    --loss CE \
    --init_eval False \
    --mode train