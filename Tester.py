from custom_transforms import transforms
#import custom_models.segmentation as tvmodels
import torch
from PIL import Image
import numpy as np
from custom_transforms import functional as func
from tqdm import tqdm
import os
from utils.metrics import Evaluator

import models
import math 

from utils.utils import ret2mask,mask2label,get_test_times

class Tester(object):
    def __init__(self, args):
        self.crop_size = args.crop_size
        self.stride = args.stride
        assert self.crop_size >= self.stride
        if args.by_trainer:
            self.init_by_trainer(args)
        else:
            self.init_by_args(args)
        self.tensor_trans = transforms.ToTensor_single()
        self.normalize_trans = transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])
        
        self.list = args.eval_list
        self.img_path = args.img_path
        self.gt_path = args.gt_path
        self.test_list = self.get_pairs()
        print(f"{len(self.test_list)} pairs to test...")
        self.evaluator = Evaluator(args.num_of_class)
        
    def init_by_trainer(self,args): 
        self.model = args.model
        self.cuda = args.cuda

    def init_by_args(self,args):
        if args.model == 'fcn':
            self.model = models.FCN8(num_classes=args.num_of_class)
        elif args.model == 'deeplabv3+':
            self.model = models.DeepLab(num_classes=args.num_of_class,backbone='resnet')
        elif args.model == 'carafe':
            self.model = models.DeepLab_CARAFE(num_classes=args.num_of_class,backbone='resnet')
        elif args.model == 'unet':
            self.model = models.UNet(num_classes=args.num_of_class)
        elif args.model == 'gcn':
            self.model = models.GCN(num_classes=args.num_of_class)
        elif args.model == 'pspnet':
            self.model = models.PSPNet(num_classes=args.num_of_class)
        else:
            raise NotImplementedError

        self.cuda = args.cuda
        if self.cuda:
            self.model = self.model.cuda()
            checkpoint = torch.load(args.checkpoint) 
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
        self.model.load_state_dict(checkpoint['parameters'])
        self.epoch = checkpoint['epoch']

    def get_pairs(self):
        test_list = [] 
        with open(self.list) as f:
            for each_file in f:
                file_name = each_file.strip()
                img = os.path.join(self.img_path,file_name)
                file_name = file_name[:-7]+"label.tif"
                gt = os.path.join(self.gt_path,file_name)
                assert os.path.isfile(img),"Images %s cannot be found!" %img
                assert os.path.isfile(gt),"Ground truth %s cannot be found!" %gt
                test_list.append((img,gt))
        return test_list

    def get_pointset(self,img):
        W, H = img.size
        pointset = []
        count=0
        i = 0
        while i<W:
            break_flag_i = False
            if i+self.crop_size >= W:
                i = W - self.crop_size
                break_flag_i = True
            j = 0
            while j<H:
                break_flag_j = False
                if j + self.crop_size >= H:
                    j = H - self.crop_size
                    break_flag_j = True
                count+=1
                pointset.append((i,j))
                if break_flag_j:
                    break
                j+=self.stride
            if break_flag_i:
                break
            i+=self.stride
        value = get_test_times(W,H,self.crop_size,self.stride)
        assert count==value,f'count={count} while get_test_times returns {value}'
        return count, pointset 

    def run(self,train_epoch=-1,save=False):
        for img_file,gt_file in self.test_list:
            print(f"Start testing {img_file}...")
            if save and os.path.exists("epoch"+str(train_epoch)) is False:
                os.mkdir("epoch"+str(train_epoch))
            self.test_one_large(img_file,gt_file,train_epoch,save)
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union() 
        print("Acc:",Acc)
        print("Acc_class:",Acc_class)
        print("mIoU:",mIoU)
        print("FWIoU:",FWIoU)
        return Acc,Acc_class,mIoU,FWIoU
        
    def test_one_large(self,img_file,gt_file,train_epoch,save):
        img = Image.open(img_file).convert('RGB')
        H, W = img.size
        gt = np.array(Image.open(gt_file))
        times, points = self.get_pointset(img)
        
        print(f'{times} tests will be carried out on {img_file}...')
        tbar = tqdm(points)
        label_map = np.zeros([H,W],dtype=np.uint8)
        score_map = np.zeros([H,W],dtype=np.uint8)
        for i,j in tbar:
            tbar.set_description(f"{i},{j}")
            label_map,score_map = self.test_patch(i,j,img,label_map,score_map)
        #finish a 6000x6000
        self.evaluator.add_batch(label_map,gt)
        
        #save mask
        if save:   
            mask = ret2mask(label_map)
            png_name = os.path.join("epoch"+str(train_epoch),os.path.basename(img_file).split('.')[0]+'.png')
            Image.fromarray(mask).save(png_name)

    def test_patch(self,i,j,img,label_map,score_map):
        cropped = func.crop(img,i,j,self.crop_size,self.crop_size)
        cropped = self.tensor_trans(cropped)
        cropped = self.normalize_trans(cropped).unsqueeze(0)
        self.model.eval()
        if self.cuda:
            cropped = cropped.cuda()
        #out = self.model(cropped)['out']
        out = self.model(cropped)
        #out = torch.nn.functional.softmax(out, dim=1)
        ret = torch.max(out.squeeze(),dim=0)
        score = ret[0].data.detach().cpu().numpy()
        label = ret[1].data.detach().cpu().numpy()

        score_temp = score_map[i:i+self.crop_size,j:j+self.crop_size]
        label_temp = label_map[i:i+self.crop_size,j:j+self.crop_size]
        index = score > score_temp
        score_temp[index] = score[index]
        label_temp[index] = label[index]
        label_map[i:i+self.crop_size,j:j+self.crop_size] = label_temp
        score_map[i:i+self.crop_size,j:j+self.crop_size] = score_temp

        return label_map,score_map
