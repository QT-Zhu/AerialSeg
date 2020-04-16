import argparse
from Trainer import Trainer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported bool string format.')

def main():
    parser = argparse.ArgumentParser(description="AerialSeg by PyTorch: train.py")
    parser.add_argument('--train_batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--train_list', type=str, default='Potsdam/Potsdam_train.txt', help='list file for training')
    parser.add_argument('--eval_list', type=str, default='Potsdam/Potsdam_val.txt', help='list file for validation')
    parser.add_argument('--img_path', type=str, default='Potsdam/2_Ortho_RGB', help='path for images of dataset')
    parser.add_argument('--gt_path', type=str, default='Potsdam/5_Labels_all', help='path for ground truth of dataset')
    parser.add_argument('--num_of_class', type=int, default=6, help='number of classes')
    parser.add_argument('--crop_size', type=int, default=512, help='crop size of input images')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume from')
    parser.add_argument('--finetune', type=str, default=None, help='checkpoint to finetune')
    parser.add_argument('--cuda', type=str2bool, default=False, help='whether to use GPU')
    parser.add_argument('--loss', type=str, default='CE', help='type of loss function')
    parser.add_argument('--model', type=str, default='deeplabv3', help='model to train')
    parser.add_argument('--schedule_mode', type=str, default='poly', help='which scheduler to apply')
    parser.add_argument('--init_eval', type=str2bool, default=False, help='whether to start with evaluation')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate of training')
    #parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size for validation')
    #parser.add_argument('--save_interval', type=int, default=10, help='frequency for save model and prediction')
    #parser.add_argument('--eval_interval', type=int, default=1, help='frequency for validation')
    
    args = parser.parse_args()
    print(args)
    my_trainer = Trainer(args)
    my_trainer.run()

if __name__ == "__main__":
   main()