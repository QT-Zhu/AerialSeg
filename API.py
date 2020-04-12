import models
import torch
import argparse
import os
from PIL import Image
from custom_transforms import transforms
from utils.utils import ret2mask
system_path = os.path.dirname(__file__)
model_zoo={
    'FCN': os.path.join(system_path,'checkpoints/FCN8_epoch137.pth.tar'),
    'DeepLab': os.path.join(system_path,'checkpoints/DeepLab_epoch240.pth.tar'),
    'GCN': os.path.join(system_path,'checkpoints/')
}



def run_FCN(img_path):
    model = models.FCN8(6)
    checkpoint = torch.load(model_zoo['FCN'], map_location='cpu')
    model.load_state_dict(checkpoint['parameters'])
    tensor_trans = transforms.ToTensor_single()
    normalize_trans = transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])
    img = Image.open(img_path).convert('RGB')
    cropped = normalize_trans(tensor_trans(img)).unsqueeze(0)
    model.eval()
    out = model(cropped)
    ret = torch.max(out.squeeze(),dim=0)
    label = ret[1].data.detach().cpu().numpy()
    mask = ret2mask(label)
    save_path = os.path.join(system_path,os.path.basename(img_path).split('.')[0]+'.png')
    Image.fromarray(mask).save(save_path)
    return save_path
    
def run_DeepLab(img_path):
    model = models.DeepLab(6,backbone='resnet')
    checkpoint = torch.load(model_zoo['DeepLab'], map_location='cpu')
    model.load_state_dict(checkpoint['parameters'])
    tensor_trans = transforms.ToTensor_single()
    normalize_trans = transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])
    img = Image.open(img_path).convert('RGB')
    cropped = normalize_trans(tensor_trans(img)).unsqueeze(0)
    model.eval()
    out = model(cropped)
    ret = torch.max(out.squeeze(),dim=0)
    label = ret[1].data.detach().cpu().numpy()
    mask = ret2mask(label)
    save_path = os.path.join(system_path,os.path.basename(img_path).split('.')[0]+'.png')
    Image.fromarray(mask).save(save_path)
    return save_path

def run_GCN(img_path):
    model = models.GCN(6)
    checkpoint = torch.load(model_zoo['GCN'], map_location='cpu')
    model.load_state_dict(checkpoint['parameters'])
    tensor_trans = transforms.ToTensor_single()
    normalize_trans = transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])
    img = Image.open(img_path).convert('RGB')
    cropped = normalize_trans(tensor_trans(img)).unsqueeze(0)
    model.eval()
    out = model(cropped)
    ret = torch.max(out.squeeze(),dim=0)
    label = ret[1].data.detach().cpu().numpy()
    mask = ret2mask(label)
    save_path = os.path.join(system_path,os.path.basename(img_path).split('.')[0]+'.png')
    Image.fromarray(mask).save(save_path)
    return save_path