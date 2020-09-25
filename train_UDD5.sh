python3 train.py \
    --cuda True \
    --epoch 300 \
    --train_batch_size 8 \
    --dataset UDD5 \
    --data_path ~/datasets/UDD5 \
    --model ENet \
    --loss CE \
    --init_eval True \
    --mode train