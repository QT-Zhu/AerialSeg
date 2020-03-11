from PIL import Image
import numpy as np
import os
from tqdm import tqdm

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







