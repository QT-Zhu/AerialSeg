from custom_transforms import transforms
import custom_models.segmentation as models
import torch

class Tester(object):
    def __init__(self, args):
        self.crop_size = args.crop_size
        self.stride = args.stride
        self.model = models.deeplabv3_resnet50(num_classes=args.num_of_class)

        self.cuda = args.cuda
        if self.cuda:
            self.model = self.model.cuda()
            checkpoint = torch.load(args.checkpoint) 
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.epoch = checkpoint['epoch']

    def run(self):
        raise NotImplementedError


class TestAug(object):
    def __init__(self,crop_size):
        self.crop_trans = transforms.RandomCrop(crop_size,)
        self.tensor_trans = transforms.ToTensor()
        self.normalize_trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    def __call__(self, img, stride):
        raise NotImplementedError