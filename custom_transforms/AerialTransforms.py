from . import transforms
from . import functional as F
#import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
class TrainAug(object):
    def __init__(self,crop_size):
        #self.crop_trans = transforms.RandomCrop(crop_size)
        self.crop_trans = transforms.ZQTRandomCrop(crop_size)
        self.hflip_trans = transforms.RandomHorizontalFlip()
        self.vflip_trans = transforms.RandomVerticalFlip()
        #self.rotate_trans = transforms.RandomRotation(degrees=[0,90,180,270])
        #Bug to be discovered
        self.tensor_trans = transforms.ToTensor()
        self.normalize_trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    def __call__(self,img,gt): #img and gt are PIL objects
        _, img ,gt = self.crop_trans(img,gt) #coordination is useless for training
        
        img, gt = self.hflip_trans(img,gt)
        img, gt = self.vflip_trans(img,gt)
        
        img, gt = self.tensor_trans(img,gt)
        img = self.normalize_trans(img)
        #gt has to transform from (batchsize,1,H,W) to (batchsize,H,W)
        #loss func will transfer it to one-hot automatically
        gt = gt.squeeze()
        return img,gt

#deprecated
class EvalAug(object):
    def __init__(self,crop_size):
        self.crop_trans = transforms.RandomCrop(crop_size)
        self.tensor_trans = transforms.ToTensor()
        self.normalize_trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    def __call__(self,img,gt):
        _, img, gt = self.crop_trans(img,gt)
        img, gt = self.tensor_trans(img,gt)
        img_tensor = self.normalize_trans(img)
        gt = gt.squeeze()
        return img, img_tensor, gt
