import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler as Disample
import torchvision
import torchvision.transforms as tt
import torchvision.transforms.functional as ttf

from .Transforms_HF import MyRandomApply

# dataset
class TrainingDataset(Dataset):
    def __init__(self, data_path, data_id_list, img_transforms=None, transforms=None, p=None, opt=None):
        self.data_id_list = data_id_list
        self.data_path = data_path
        self.len = len(self.data_id_list)
        self.img_transforms = img_transforms
        self.transforms = transforms
        self.p = p
        self.opt = opt

        # mean and std for training dataset
        # self.mean = torch.tensor([0.3429, 0.3525, 0.3516])
        # self.std = torch.tensor([0.1836, 0.1878, 0.1868])

        # self.mean = torch.tensor([0.0, 0.0, 0.0])
        # self.std = torch.tensor([1.0, 1.0, 1.0])

        # self.normalize=torchvision.transforms.Normalize(mean=self.mean,std=self.std)
        self.img_dict, self.mask_dict = {},{}
        for case in data_id_list:
            self.img_dict[case] = case + '.png'
            self.mask_dict[case] = case + '-mask.png'

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        data_idx = self.data_id_list[idx]
        img = Image.open(os.path.join(self.data_path, self.img_dict[data_idx])).convert('RGB')
        label = Image.open(os.path.join(self.data_path, self.mask_dict[data_idx])).convert('L')

        if self.img_transforms is not None: 
            # apply transforms on imgs, tt.RandomApply([tt.ColorJitter(0.1, 0.1, 0.1, 0.1), MyRandGamma(r=0.1)])
            img = self.img_transforms(img)

        objects = [img,label]
        if self.transforms is not None:
            # only done for training case
            objects = MyRandomApply(objects, self.transforms, self.p)

        img = ttf.to_tensor(objects[0])
        # img = self.normalize(img)

        labels = ttf.to_tensor(objects[1])>0.0001

        file_id = self.data_id_list[idx]
        return img.float(), labels.float(), file_id
        # return {'data': img.float(),
        #         'label': labels,
        #         'file_id': file_id}


# dataloader
def dataloader(data_path, data_id_list, batch_size=10,shuffle=False,num_workers=0,img_transforms=None,transforms=None,p=None,opt=None):
    dataset = TrainingDataset(data_path, data_id_list, img_transforms, transforms, p, opt)
    try:
        Sampler = Disample(dataset,num_replicas=opt.world_size,rank=opt.rank,shuffle=shuffle)
        dl = DataLoader(dataset,batch_size=batch_size,sampler=Sampler,num_workers=num_workers,pin_memory=True)
    except:
        dl = DataLoader(dataset,batch_size = batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    return dl


# generate train, val dataloader
def get_dataloaders(opt):
    # import training_id and validation_id list from txt file
    training_id, validation_id = [], []
    with open(opt.train_id_loc,'r') as file:
        for line in file:
            training_id.append(line.split('.')[0])

    with open(opt.val_id_loc,'r') as file:
        for line in file:
            validation_id.append(line.split('.')[0])

    training_id.sort()
    validation_id.sort()

    # define transformation for imgs and labels
    img_transforms, transforms, p = None, None, None
    if opt.augmentation and opt.train_or_test == "train":
        from .Transforms_HF import MyRandResizedCrop, MyHFlip, MyRandRotate, MyVFlip, MyRandGamma
        # scale = (0.95, 1)
        # ratio = (5. / 6., 6. / 5.)
        # RandResizedCrop = MyRandResizedCrop(scale, ratio)
        transforms = [MyHFlip, MyRandRotate, MyVFlip]
        p = [0.3 for x in transforms]
        img_transforms = tt.RandomApply([tt.ColorJitter(0.1, 0.1, 0.1, 0.1), MyRandGamma(r=0.1)])
        if opt.rank==0:
            print('Augmentation added!')

    train_dataloader = dataloader(opt.data_path, training_id, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, img_transforms=img_transforms, transforms=transforms, p=p, opt=opt)

    test_dataloader = dataloader(opt.data_path, validation_id, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, opt=opt)

    return train_dataloader, test_dataloader

'''
ColorJitter:
Randomly change the brightness, contrast, saturation and hue of an image. 
https://pytorch.org/vision/master/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
https://pytorch.org/vision/master/transforms.html?highlight=colorjitter#torchvision.transforms.ColorJitter
'''