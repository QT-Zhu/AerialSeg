from torch.utils.data.dataloader import DataLoader
from AerialDataset import AerialDataset
import torch
import os
import custom_models.segmentation as models
import torch.nn as nn
from utils import ret2mask
import matplotlib.pyplot as plt
from metrics import Evaluator
import numpy as np
from PIL import Image

#For Lovasz loss
from utils import LovaszSoftmax


class Trainer(object):
    def __init__(self, args):
        self.epochs = args.epochs
        self.save_interval = args.save_interval
        self.train_data = AerialDataset(args,mode='train')
        self.train_loader =  DataLoader(self.train_data,batch_size=args.train_batch_size,shuffle=True,
                          num_workers=1)
        self.model = models.deeplabv3_resnet50(num_classes=args.num_of_class)

        self.loss = args.loss
        if self.loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        else: #self.loss == 'LS'
            self.criterion = LovaszSoftmax()

        self.optimizer = torch.optim.AdamW(self.model.parameters())

        self.eval_interval = args.eval_interval
        self.eval_data = AerialDataset(args,mode='eval')
        self.eval_loader = DataLoader(self.eval_data,batch_size=args.eval_batch_size,shuffle=False,
                          num_workers=1)
        self.evaluator = Evaluator(args.num_of_class)
        
        self.cuda = args.cuda
        if self.cuda is True:
            self.model = self.model.cuda()

        self.resume = args.resume
        if self.resume != None:
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['opt'])
            self.start_epoch = checkpoint['epoch'] + 1
            #start from next epoch
        else:
            self.start_epoch = 1
    
    def test(self):
        raise NotImplementedError

    def run(self):
        end_epoch = self.start_epoch + self.epochs
        for epoch in range(self.start_epoch,end_epoch):
            self.train(epoch)
            if (epoch-self.start_epoch+1)%self.save_interval==0:
                saved_dict = {'epoch': epoch,'model': self.model.state_dict(),'opt': self.optimizer.state_dict()}
                torch.save(saved_dict, './model_epoch'+str(epoch)+'.pth.tar')
            if (epoch-self.start_epoch+1)%self.eval_interval==0:
                self.eval(epoch)

    def train(self,epoch):
        self.model.train()
        print(f"----------epoch {epoch}----------")
        for i,[img,gt] in enumerate(self.train_loader):
            print("epoch:",epoch," batch:",i+1)
            print("img:",img.shape)
            print("gt:",gt.shape)
            self.optimizer.zero_grad()
            if self.cuda:
                img,gt = img.cuda(),gt.cuda()
            pred = self.model(img)['out']
            print("pred:",pred.shape)
            loss = self.criterion(pred,gt.long())
            print("loss:",loss)
            loss.backward()
            self.optimizer.step()

    def eval(self,epoch):
        self.model.eval()
        self.evaluator.reset()
        os.mkdir("epoch"+str(epoch))
        print(f"-----eval epoch {epoch}-----")
        for i,[img,gt] in enumerate(self.eval_loader):
            print("batch:",i)
            print("img:",img.shape)
            print("gt:",gt.shape)
            if self.cuda:
                img = img.cuda()
            out = self.model(img)['out']
            pred = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
            print("pred:",pred.shape)
            gt = gt.numpy().squeeze()
            #Both gt and pred are numpy array now
            self.evaluator.add_batch(gt,pred)

            #colorise
            img = img.cpu().numpy()
            mask = ret2mask(pred)
            gt_color = ret2mask(gt)
            cat = np.concatenate((gt_color,img,mask),axis=1)
            cat = Image.fromarray(cat)
            cat.save("epoch"+str(epoch)+"/batch"+str(i)+".png")
        
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        
        print("Acc:",Acc)
        print("Acc_class:",Acc_class)
        print("mIoU:",mIoU)
        print("FWIoU:",FWIoU)


if __name__ == "__main__":
   print("--Trainer.py--test--")
   
