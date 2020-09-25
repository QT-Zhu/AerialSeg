from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import math


def get_test_times(width, height, crop_size, stride):
    x = math.ceil((width - crop_size)/stride) + 1
    y = math.ceil((height - crop_size)/stride) + 1
    return x*y


#numpy2numpy
def mask2label(mask,dataset):
    rgb2label={}
    if dataset=='Potsdam':
        rgb2label[(255,255,255)]=0 # Surfaces
        rgb2label[(0,0,255)]=1     # Building
        rgb2label[(0,255,255)]=2   # Low vegetation
        rgb2label[(0,255,0)]=3     # Tree
        rgb2label[(255,255,0)]=4   # Car
        rgb2label[(255,0,0)]=5     # Clutter / background
    elif dataset=='UDD5':
        rgb2label[(107,142,35)]=0     # Vegetation
        rgb2label[(102,102,156)]=1    # Building
        rgb2label[(128,64,128)]=2     # Road
        rgb2label[(0,0,142)]=3        # Vehicle
        rgb2label[(0,0,0)]=4          # Other
    elif dataset=='UDD6':
        rgb2label[(0,0,0)]=0          # Other
        rgb2label[(102,102,156)]=1    # Building
        rgb2label[(128,64,128)]=2     # Road
        rgb2label[(107,142,35)]=3     # Vegetation
        rgb2label[(0,0,142)]=4        # Vehicle
        rgb2label[(70,70,70)]=5       # Roof	
    else:
        raise NotImplementedError
    label_map = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)
    for rgb in rgb2label:
        color = [rgb[0],rgb[1],rgb[2]]
        label_map[np.where(np.all(mask == color, axis=-1))[:2]] = rgb2label[rgb]
    return label_map

#numpy2numpy
def ret2mask(pred,dataset):
    label2rgb={}
    if dataset=='Potsdam':
        label2rgb[0]=(255,255,255)
        label2rgb[1]=(0,0,255)
        label2rgb[2]=(0,255,255)
        label2rgb[3]=(0,255,0)
        label2rgb[4]=(255,255,0)
        label2rgb[5]=(255,0,0)
    elif dataset=='UDD5':
        label2rgb[0]=(107,142,35)
        label2rgb[1]=(102,102,156)
        label2rgb[2]=(128,64,128)
        label2rgb[3]=(0,0,142)
        label2rgb[4]=(0,0,0)
    elif dataset=='UDD6':
        label2rgb[0]=(0,0,0)
        label2rgb[1]=(102,102,156)
        label2rgb[2]=(128,64,128)
        label2rgb[3]=(107,142,35)
        label2rgb[4]=(0,0,142)
        label2rgb[5]=(70,70,70)
    else:
        raise NotImplementedError
    mask = np.zeros((pred.shape[0],pred.shape[1],3),dtype=np.uint8)
    for label in label2rgb:
        index = np.where(pred == label)[:2]
        mask[index] = label2rgb[label]
    return mask

def tensor2PIL(tensor):
    return None


if __name__ == '__main__':
    src='UDD6/val/gt'
    dest='UDD6/val/gt_vis'
    print("Transfer from "+src+"\nto "+dest)
    if not (os.path.isdir(dest)):
        os.mkdir(dest)
    for filename in tqdm(os.listdir(src)):
        if filename[0]=='.':
            continue
        img = Image.open(os.path.join(src,filename)).convert('RGB')
        img = np.array(img)
        ret = ret2mask(img,dataset='UDD6')
        ret = Image.fromarray(ret)
        ret.save(os.path.join(dest,filename))











