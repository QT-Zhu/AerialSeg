from torch.utils.data.dataset import Dataset
import os
from PIL import Image
#from .utils import mask2label
from custom_transforms import AerialTransforms



class AerialDataset(Dataset):
    def __init__(self,args,mode):
        self.mode = mode
        self.crop_size = args.crop_size
        self.img_path, self.gt_path= os.path.join(os.getcwd(),args.img_path),os.path.join(os.getcwd(),args.gt_path)
        self.img_list,self.gt_list = [],[]
        if self.mode=='train':
            self.list = args.train_list
        else: #deprecated: self.mode == 'eval'
            self.list = args.eval_list
        with open(self.list) as f:
            for each_file in f:
                file_name = each_file.strip()
                img = os.path.join(self.img_path,file_name)
                file_name = file_name[:-7]+"label.tif"
                gt = os.path.join(self.gt_path,file_name)
                assert os.path.isfile(img),"Images %s cannot be found!" %img
                assert os.path.isfile(gt),"Ground truth %s cannot be found!" %gt
                self.img_list.append(img)
                self.gt_list.append(gt)
        print(f"{len(self.img_list)} pairs to {self.mode}")
        if self.mode == 'train':
            self.augtrans = AerialTransforms.TrainAug(self.crop_size)
        else: #self.mode == 'eval'
            self.augtrans = AerialTransforms.EvalAug(self.crop_size)
        
    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self,index):
        img = Image.open(self.img_list[index]).convert('RGB')
        #gt = mask2label(Image.open(self.gt_list[index])) 
        
        #or more efficiently, directly load label map
        gt = Image.open(self.gt_list[index])
        #Trans from PIL pair to tensor pair
        return self.augtrans(img,gt)
        
if __name__ == "__main__":
   print("AerialDataset.py")
