'''
from . import functional as F
#import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
'''
from .transforms import RandomCrop,RandomRotate,HorizontalFlip,VerticalFlip,ToTensor,Normalize
import numpy as np
from torchvision import transforms


class TrainAug(object):
    def __init__(self,crop_size):
        self.trans = transforms.Compose([
            RandomCrop(crop_size),
            transforms.RandomChoice([
                RandomRotate([90,270]),
                HorizontalFlip(),
                VerticalFlip()
            ]),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()      
        ])
        
    
    def __call__(self,img,gt): #img and gt are PIL objects
        sample = {'image':img, 'label':gt}
        sample = self.trans(sample)

        #gt has to transform from (batchsize,1,H,W) to (batchsize,H,W)
        #loss func will transfer it to one-hot automatically
        img = sample['image']
        gt = sample['label']
        gt = gt.squeeze(1)
        return img,gt

class EvalAug(object):
    def __init__(self):
        pass

if __name__=='__main__':
    np.random.seed(0)
    from PIL import Image
    img = Image.open("../UDD5/val/src/000061.JPG").convert('RGB')
    mask = Image.open("../UDD5/val/gt/000061.png")
    print(img.size)
    t = TrainAug(1000)
    img,mask = t(img,mask)
    print(img.shape,mask.shape)
    print(img)
    print(mask)
    
