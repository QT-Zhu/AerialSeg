from PIL import Image
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight 
from lovasz_losses import lovasz_softmax

def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=None):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


# surfaces(RGB: 255, 255, 255)
# Building(RGB: 0, 0, 255)
# Low vegetation(RGB: 0, 255, 255)
# Tree(RGB: 0, 255, 0)
# Car(RGB: 255, 255, 0)
# Clutter / background(RGB: 255, 0, 0)

# From PIL to PIL
def mask2label(mask):
    rgb2label={}
    rgb2label[(255,255,255)]=0
    rgb2label[(0,0,255)]=1
    rgb2label[(0,255,255)]=2
    rgb2label[(0,255,0)]=3
    rgb2label[(255,255,0)]=4
    rgb2label[(255,0,0)]=5
    mask = np.array(mask) # mask is a PIL object
    label_map = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)

    for rgb in rgb2label:
        color = [rgb[0],rgb[1],rgb[2]]
        label_map[np.where(np.all(mask == color, axis=-1))[:2]] = rgb2label[rgb]
    return Image.fromarray(label_map)

def ret2mask(pred):
    label2rgb={}
    label2rgb[0]=(255,255,255)
    label2rgb[1]=(0,0,255)
    label2rgb[2]=(0,255,255)
    label2rgb[3]=(0,255,0)
    label2rgb[4]=(255,255,0)
    label2rgb[5]=(255,0,0)
    mask = np.zeros((pred.shape[0],pred.shape[1],3),dtype=np.uint8)
    for label in label2rgb:
        index = np.where(pred == label)[:2]
        mask[index] = label2rgb[label]
    return mask

def tensor2PIL(tensor):
    return None

def main(src,dest):
    print("Transfer from "+src+"\nto "+dest)
    if not (os.path.isdir(dest)):
        os.mkdir(dest)
    for filename in tqdm(os.listdir(src)):
        if filename[0]=='.':
            continue
        img = Image.open(os.path.join(src,filename)).convert('RGB')
        ret = mask2label(img)
        ret.save(os.path.join(dest,filename))



#main('Potsdam/5_Labels_all','Potsdam/Potsdam_label')







