from torch.utils.data.dataloader import DataLoader
from utils.AerialDataset import AerialDataset
import torch
import os
import torch.nn as nn
import torch.optim as opt
from utils.utils import ret2mask,get_test_times
import matplotlib.pyplot as plt
from utils.metrics import Evaluator
import numpy as np
from PIL import Image

#For global test
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter
import torchvision.transforms.functional as F
from utils.transforms import EvaluationTransform


#For loss and scheduler
from utils.loss import CE_DiceLoss, CrossEntropyLoss2d, LovaszSoftmax, FocalLoss
from utils.scheduler import Poly, OneCycle
import models

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.epochs = args.epochs
        self.dataset = args.dataset
        self.data_path = args.data_path
        self.train_crop_size = args.train_crop_size
        self.eval_crop_size = args.eval_crop_size
        self.stride = args.stride
        self.batch_size = args.train_batch_size
        self.train_data = AerialDataset(crop_size=self.train_crop_size,dataset=self.dataset,data_path=self.data_path,mode='train')
        self.train_loader =  DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True,
                          num_workers=2)
        self.eval_data = AerialDataset(dataset=self.dataset,data_path=self.data_path,mode='val')
        self.eval_loader = DataLoader(self.eval_data,batch_size=1,shuffle=False,num_workers=2)

        if self.dataset=='Potsdam':
            self.num_of_class=6
            self.epoch_repeat = get_test_times(6000,6000,self.train_crop_size,self.train_crop_size)
        elif self.dataset=='UDD5':
            self.num_of_class=5
            self.epoch_repeat = get_test_times(4000,3000,self.train_crop_size,self.train_crop_size)
        elif self.dataset=='UDD6':
            self.num_of_class=6
            self.epoch_repeat = get_test_times(4000,3000,self.train_crop_size,self.train_crop_size)
        else:
            raise NotImplementedError

        if args.model == 'FCN':
            self.model = models.FCN8(num_classes=self.num_of_class)
        elif args.model == 'DeepLabV3+':
            self.model = models.DeepLab(num_classes=self.num_of_class,backbone='resnet')
        elif args.model == 'GCN':
            self.model = models.GCN(num_classes=self.num_of_class)
        elif args.model == 'UNet':
            self.model = models.UNet(num_classes=self.num_of_class)
        elif args.model == 'ENet':
            self.model = models.ENet(num_classes=self.num_of_class)
        elif args.model == 'D-LinkNet':
            self.model = models.DinkNet34(num_classes=self.num_of_class)
        else:
            raise NotImplementedError

        if args.loss == 'CE':
            self.criterion = CrossEntropyLoss2d()
        elif args.loss == 'LS':
            self.criterion = LovaszSoftmax()
        elif args.loss == 'F':
            self.criterion = FocalLoss()
        elif args.loss == 'CE+D':
            self.criterion = CE_DiceLoss()
        else:
            raise NotImplementedError
        
        self.schedule_mode = args.schedule_mode
        self.optimizer = opt.AdamW(self.model.parameters(),lr=args.lr)
        if self.schedule_mode == 'step':
            self.scheduler = opt.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.schedule_mode == 'miou' or self.schedule_mode == 'acc':
            self.scheduler = opt.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=10, factor=0.1)
        elif self.schedule_mode == 'poly':
            iters_per_epoch=len(self.train_loader)
            self.scheduler = Poly(self.optimizer,num_epochs=args.epochs,iters_per_epoch=iters_per_epoch)
        else:
            raise NotImplementedError

        self.evaluator = Evaluator(self.num_of_class)

        self.model = nn.DataParallel(self.model)
        
        self.cuda = args.cuda
        if self.cuda is True:
            self.model = self.model.cuda()

        self.resume = args.resume
        self.finetune = args.finetune
        assert not (self.resume != None and self.finetune != None)

        if self.resume != None:
            print("Loading existing model...")
            if self.cuda:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu') 
            self.model.load_state_dict(checkpoint['parameters'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            #start from next epoch
        elif self.finetune != None:
            print("Loading existing model...")
            if self.cuda:
                checkpoint = torch.load(args.finetune)
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')
            self.model.load_state_dict(checkpoint['parameters'])
            self.start_epoch = checkpoint['epoch'] + 1
        else:
            self.start_epoch = 1
        if self.mode=='train':    
            self.writer = SummaryWriter(comment='-'+self.dataset+'_'+self.model.__class__.__name__+'_'+args.loss)
        self.init_eval = args.init_eval
        
    #Note: self.start_epoch and self.epochs are only used in run() to schedule training & validation
    def run(self):
        if self.init_eval: #init with an evaluation
            init_test_epoch = self.start_epoch - 1
            Acc,_,mIoU,_ = self.validate(init_test_epoch,save=True)
            self.writer.add_scalar('eval/Acc', Acc, init_test_epoch)
            self.writer.add_scalar('eval/mIoU', mIoU, init_test_epoch)
            self.writer.flush()
        end_epoch = self.start_epoch + self.epochs
        for epoch in range(self.start_epoch,end_epoch):  
            loss = self.train(epoch)
            self.writer.add_scalar('train/lr',self.optimizer.state_dict()['param_groups'][0]['lr'],epoch)
            self.writer.add_scalar('train/loss',loss,epoch)
            self.writer.flush()
            saved_dict = {
                'model': self.model.__class__.__name__,
                'epoch': epoch,
                'dataset': self.dataset,
                'parameters': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }
            torch.save(saved_dict, f'./{self.model.__class__.__name__}_{self.dataset}_epoch{epoch}.pth.tar')
            
            Acc,_,mIoU,_ = self.validate(epoch,save=True)
            self.writer.add_scalar('eval/Acc',Acc,epoch)
            self.writer.add_scalar('eval/mIoU',mIoU,epoch)
            self.writer.flush()
            if self.schedule_mode == 'step' or self.schedule_mode == 'poly':
                self.scheduler.step()
            elif self.schedule_mode == 'miou':
                self.scheduler.step(mIoU)
            elif self.schedule_mode == 'acc':
                self.scheduler.step(Acc)
            else:
                raise NotImplementedError
        self.writer.close()

    def train(self,epoch):
        self.model.train()
        print(f"----------epoch {epoch}----------")
        print("lr:",self.optimizer.state_dict()['param_groups'][0]['lr'])
        total_loss = 0
        num_of_batches = len(self.train_loader) * self.epoch_repeat
        for itr in range(100):
            for i,[img,gt] in enumerate(self.train_loader):
                print(f"epoch: {epoch} batch: {i+1+itr*len(self.train_loader)}/{num_of_batches}")
                print("img:",img.shape)
                print("gt:",gt.shape)
                self.optimizer.zero_grad()
                if self.cuda:
                    img,gt = img.cuda(),gt.cuda()
                pred = self.model(img)
                print("pred:",pred.shape)
                loss = self.criterion(pred,gt.long())
                print("loss:",loss)
                total_loss += loss.data
                loss.backward()
                self.optimizer.step()
        return total_loss

    def validate(self,epoch,save):
        self.model.eval()
        print(f"----------validate epoch {epoch}----------")
        if save and not os.path.exists("epoch_"+str(epoch)):
            os.mkdir("epoch"+str(epoch))
        num_of_imgs = len(self.eval_loader)
        for i,sample in enumerate(self.eval_loader):
            img_name,gt_name = sample['img'][0],sample['gt'][0]
            print(f"{i+1}/{num_of_imgs}:")

            img = Image.open(img_name).convert('RGB')
            gt = np.array(Image.open(gt_name))
            times, points = self.get_pointset(img)
            print(f'{times} tests will be carried out on {img_name}...')
            W,H = img.size #TODO: check numpy & PIL dimensions
            label_map = np.zeros([H,W],dtype=np.uint8)
            score_map = np.zeros([H,W],dtype=np.uint8)
            tbar = tqdm(points)
            for i,j in tbar:
                tbar.set_description(f"{i},{j}")
                label_map,score_map = self.test_patch(i,j,img,label_map,score_map)
            #finish a large
            self.evaluator.add_batch(label_map,gt)
            if save:   
                mask = ret2mask(label_map,dataset=self.dataset)
                png_name = os.path.join("epoch"+str(epoch),os.path.basename(img_name).split('.')[0]+'.png')
                Image.fromarray(mask).save(png_name)
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union() 
        print("Acc:",Acc)
        print("Acc_class:",Acc_class)
        print("mIoU:",mIoU)
        print("FWIoU:",FWIoU)
        return Acc,Acc_class,mIoU,FWIoU

    def test_patch(self,i,j,img,label_map,score_map):
        tr = EvaluationTransform(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])
        #print(img.size)
        cropped = img.crop((i,j,i+self.eval_crop_size,j+self.eval_crop_size))
        cropped = tr(cropped).unsqueeze(0)
        if self.cuda:
            cropped = cropped.cuda()
        out = self.model(cropped)
        #out = torch.nn.functional.softmax(out, dim=1)
        ret = torch.max(out.squeeze(),dim=0)
        score = ret[0].data.detach().cpu().numpy()
        label = ret[1].data.detach().cpu().numpy()

        #numpy array's shape is [H,W] while PIL.Image is [W,H]
        score_temp = score_map[j:j+self.eval_crop_size,i:i+self.eval_crop_size]
        label_temp = label_map[j:j+self.eval_crop_size,i:i+self.eval_crop_size]
        index = score > score_temp
        score_temp[index] = score[index]
        label_temp[index] = label[index]
        label_map[j:j+self.eval_crop_size,i:i+self.eval_crop_size] = label_temp
        score_map[j:j+self.eval_crop_size,i:i+self.eval_crop_size] = score_temp

        return label_map,score_map

    def get_pointset(self,img):
        W, H = img.size
        pointset = []
        count=0
        i = 0
        while i<W:
            break_flag_i = False
            if i+self.eval_crop_size >= W:
                i = W - self.eval_crop_size
                break_flag_i = True
            j = 0
            while j<H:
                break_flag_j = False
                if j + self.eval_crop_size >= H:
                    j = H - self.eval_crop_size
                    break_flag_j = True
                count+=1
                pointset.append((i,j))
                if break_flag_j:
                    break
                j+=self.stride
            if break_flag_i:
                break
            i+=self.stride
        value = get_test_times(W,H,self.eval_crop_size,self.stride)
        assert count==value,f'count={count} while get_test_times returns {value}'
        return count, pointset     




if __name__ == "__main__":
   print("--Trainer.py--")
   
