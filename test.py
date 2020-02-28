import argparse
from AerialDataset import AerialDataset
from Tester import Tester

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported bool string format.')

def main():
    parser = argparse.ArgumentParser(description="AerialSeg by PyTorch: test.py")
    #parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size for validation')
    parser.add_argument('--eval_list', type=str, default='Potsdam/Potsdam_val.txt', help='list file for validation')
    parser.add_argument('--img_path', type=str, default='Potsdam/2_Ortho_RGB', help='path for images of dataset')
    parser.add_argument('--gt_path', type=str, default='Potsdam/5_Labels_all', help='path for ground truth of dataset')
    parser.add_argument('--num_of_class', type=int, default=6, help='number of classes')
    parser.add_argument('--crop_size', type=int, default=512, help='crop size of input images')
    parser.add_argument('--stride', type=int, default=256, help='stride to test tiles')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint to test')
    parser.add_argument('--cuda', type=str2bool, default=False, help='whether to use GPU')
    parser.add_argument('--model', type=str, default='deeplab', help='model to test')
    parser.add_argument('--by_trainer', type=str2bool, default=False, help='whether called by Trainer.py')

    args = parser.parse_args()
    print(args)
    my_tester = Tester(args)
    _,_,_,_ = my_tester.run()

if __name__ == "__main__":
   main()