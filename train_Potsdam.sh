python3 train.py \
    --cuda True \
    --epoch 300 \
    --train_batch_size 8 \
    --dataset Potsdam \
    --data_path ~/datasets/Potsdam \
    --model GCN \
    --loss CE \
    --init_eval False \
    --mode train