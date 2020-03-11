from torch.utils.data.dataloader import DataLoader
from utils.AerialDataset import AerialDataset
import torch
import os
import custom_models.segmentation as tvmodels
import torch.nn as nn
import torch.optim as opt
from utils.utils import ret2mask
import matplotlib.pyplot as plt
from utils.metrics import Evaluator
import numpy as np
from PIL import Image

#For global test
from Tester import Tester
import argparse
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

#For Lovasz loss
from utils.loss import LovaszSoftmax

import models

class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.epochs = args.epochs
        
        self.train_data = AerialDataset(args,mode='train')
        self.train_loader =  DataLoader(self.train_data,batch_size=args.train_batch_size,shuffle=True,
                          num_workers=2)
        if args.model == 'fcn':
            #self.model = models.FCN8(num_classes=args.num_of_class)
            self.model = tvmodels.fcn_resnet50(num_classes=args.num_of_class)
        elif args.model == 'deeplabv3':
            self.model = tvmodels.deeplabv3_resnet50(num_classes=args.num_of_class)
        elif args.model == 'deeplabv3+':
            self.model = models.DeepLab(num_classes=args.num_of_class)
        elif args.model == 'unet':
            self.model = models.UNet(num_classes=args.num_of_class)
        elif args.model == 'pspnet':
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.loss = args.loss
        if self.loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss == 'LS':
            self.criterion = LovaszSoftmax()
        else:
            raise NotImplementedError

        self.schedule_mode = args.schedule_mode
        self.optimizer = opt.AdamW(self.model.parameters(),lr=0.05)
        if self.schedule_mode == 'step':
            self.scheduler = opt.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.schedule_mode == 'miou' or self.schedule_mode == 'acc':
            self.scheduler = opt.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=10, factor=0.1)
        else:
            raise NotImplementedError

        #self.eval_data = AerialDataset(args,mode='eval')
        #self.eval_loader = DataLoader(self.eval_data,batch_size=args.eval_batch_size,shuffle=False,num_workers=1)
        self.evaluator = Evaluator(args.num_of_class)
        
        self.cuda = args.cuda
        if self.cuda is True:
            self.model = self.model.cuda()

        self.resume = args.resume
        if self.resume != None:
            if self.cuda:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu') 
            self.model.load_state_dict(checkpoint['parameters'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            #start from next epoch
        else:
            self.start_epoch = 1
        self.writer = SummaryWriter(comment='-'+args.model+'_'+args.loss)
        self.init_eval = args.init_eval
        
    #Note: self.start_epoch and self.epochs are only used in run() to schedule training & validation
    def run(self):
        if self.init_eval: #init with an evaluation
            init_test_epoch = self.start_epoch - 1
            Acc,_,mIoU,_ = self.eval_complete(init_test_epoch,True)
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
                'parameters': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }
            torch.save(saved_dict, f'./{self.model.__class__.__name__}_epoch{epoch}.pth.tar')
            
            Acc,_,mIoU,_ = self.eval_complete(epoch)
            self.writer.add_scalar('eval/Acc',Acc,epoch)
            self.writer.add_scalar('eval/mIoU',mIoU,epoch)
            self.writer.flush()
            if self.schedule_mode == 'step':
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
        num_of_miniepoch = 144
        miniepoch_size = len(self.train_loader)
        print("#batches:",miniepoch_size*num_of_miniepoch)
        for each_miniepoch in range(num_of_miniepoch):
            for i,[img,gt] in enumerate(self.train_loader):
                print("epoch:",epoch," batch:",miniepoch_size*each_miniepoch+i+1)
                #print("img:",img.shape)
                #print("gt:",gt.shape)
                self.optimizer.zero_grad()
                if self.cuda:
                    img,gt = img.cuda(),gt.cuda()
                pred = self.model(img)['out']
                #print("pred:",pred.shape)
                loss = self.criterion(pred,gt.long())
                print("loss:",loss)
                total_loss += loss.data
                loss.backward()
                self.optimizer.step()
        return total_loss

    #deprecated in latest version
    '''
    def eval(self,epoch,save_flag):
        self.model.eval()
        self.evaluator.reset()
        if os.path.exists("epoch"+str(epoch)) is False and save_flag:
            os.mkdir("epoch"+str(epoch))
        print(f"-----eval epoch {epoch}-----")
        for i,[ori,img,gt] in enumerate(self.eval_loader):
            print("batch:",i+1)
            print("img:",img.shape)
            print("gt:",gt.shape)
            eval_batch_size = gt.shape[0]
            if self.cuda:
                img = img.cuda()
            out = self.model(img)['out']
            if eval_batch_size==1:
                pred = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
            else:
                pred = torch.argmax(out.squeeze(), dim=1).detach().cpu().numpy()
            print("pred:",pred.shape)
            gt = gt.numpy().squeeze()
            #Both gt and pred are numpy array now
            self.evaluator.add_batch(gt,pred)
            
            if save_flag:
                #colorise
                ori = ori.numpy().squeeze()
                if eval_batch_size==1:
                    mask = ret2mask(pred)
                    gt_color = ret2mask(gt)
                    ori_single = ori.transpose(1,2,0)            
                    cat = np.concatenate((gt_color,ori_single,mask),axis=1)
                    cat = Image.fromarray(np.uint8(cat))
                    cat.save("epoch"+str(epoch)+"/batch"+str(i+1)+".png")
                else:
                    for each_index in range(eval_batch_size):
                        mask = ret2mask(pred[each_index])
                        gt_color = ret2mask(gt[each_index])
                        ori_single = ori[each_index].transpose(1,2,0)
                        cat = np.concatenate((gt_color,ori_single,mask),axis=1)
                        cat = Image.fromarray(np.uint8(cat))
                        cat.save("epoch"+str(epoch)+"/batch"+str(i+1)+"_"+str(each_index)+".png")
      
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        
        print("Acc:",Acc)
        print("Acc_class:",Acc_class)
        print("mIoU:",mIoU)
        print("FWIoU:",FWIoU)

        return Acc,Acc_class,mIoU,FWIoU
    '''
    
    def eval_complete(self,epoch,save_flag=True):
        args = argparse.Namespace()
        args.by_trainer = True
        args.crop_size = self.args.crop_size
        args.stride = self.args.crop_size//2
        args.cuda = self.args.cuda
        args.model = self.model
        args.eval_list = self.args.eval_list
        args.img_path = self.args.img_path
        args.gt_path = self.args.gt_path
        args.num_of_class = self.args.num_of_class
        tester = Tester(args)
        Acc,Acc_class,mIoU,FWIoU=tester.run(train_epoch=epoch,save=save_flag)
        return Acc,Acc_class,mIoU,FWIoU

if __name__ == "__main__":
   print("--Trainer.py--")
   
