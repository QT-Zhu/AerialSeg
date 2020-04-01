python3 train.py --train_batch_size 4 --gt_path Potsdam/Potsdam_label/ --model carafe \
--loss CE --finetune ./checkpoints/DeepLab_epoch240.pth.tar --init_eval True