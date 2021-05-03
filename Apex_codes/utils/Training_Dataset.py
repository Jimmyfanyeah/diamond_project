# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:44:18 2019

@author: Who?

We load C_Labels.

Now we normalize the data
"""
import torchvision
from torch.utils.data import Dataset
from .Transforms_HF import MyRandomApply
import torchvision.transforms.functional as ttf
from PIL import Image
import os
import torch
from skimage import color

class TrainingDataset(Dataset):
    # data_list should be a list containg file names of all images
    
    def __init__(self, data_dir, data_id_list, img_transforms=None, transforms=None, p=None, opt=None):
        self.data_id_list = data_id_list
        self.data_dir = data_dir
        self.len = len(self.data_id_list)
        self.img_transforms = img_transforms
        self.transforms = transforms
        self.p = p
        self.opt = opt

    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):
        data_idx = self.data_id_list[idx]

        img_dict, mask_dict = {}, {}
        img_txt = open(self.opt.id_loc,'r')
        for img_ele in img_txt:
            img_dict[img_ele[:-5]] = img_ele[:-1]
        img_txt.close()
        mask_txt = open(self.opt.msk_id_loc,'r')
        for mask_ele in mask_txt:
            mask_dict[mask_ele[:-10]] = mask_ele[:-1]
            # mask_dict[mask_ele[:-5]] = mask_ele[:-1]
        mask_txt.close()

        # img = color.rgba2rgb(Image.open(os.path.join(self.data_dir, img_dict[data_idx])).resize((self.opt.img_size,self.opt.img_size)))
        img = Image.open(os.path.join(self.data_dir, img_dict[data_idx])).resize((self.opt.img_size, self.opt.img_size))
        labels = Image.open(os.path.join(self.data_dir, mask_dict[data_idx])).resize((self.opt.img_size,self.opt.img_size)).convert("L")

        if self.img_transforms is not None: 
            img = self.img_transforms(img)
        
        if self.transforms is not None:
            img, labels = MyRandomApply(img,labels,self.transforms, self.p)
        
        img = ttf.to_tensor(img)
        img_min = img.min()
        img_max = img.max()
        if img_max-img_min>0.00001:
            img = img.sub_(img_min).div_(img_max-img_min)

        # Only one forground class
        labels = torch.ceil(ttf.to_tensor(labels))

        file_id = self.data_id_list[idx]
        return {'data': img.float(),
                # 'labels': torch.cat(tuple(ttf.to_tensor(l) for l in labels),dim=0),
                'labels': labels.float(),
                'file_id': file_id}


class TrainingDataset_V2(TrainingDataset):
    def __init__(self, data_dir, data_id_list, img_transforms=None, transforms=None, p=None, opt=None):
        super().__init__(data_dir, data_id_list, img_transforms=img_transforms,transforms=transforms, p=p, opt=opt)

        # all data
        self.mean = torch.tensor([0.2975, 0.3034, 0.2980])
        self.std = torch.tensor([0.2735, 0.2786, 0.2750])

        # IG
        # self.mean = torch.tensor([0.3403, 0.3509, 0.3520])
        # self.std = torch.tensor([0.2464, 0.2517, 0.2493])

        # self.mean = torch.tensor([0.0, 0.0, 0.0])
        # self.std = torch.tensor([1.0, 1.0, 1.0])

        self.normalize=torchvision.transforms.Normalize(mean=self.mean,std=self.std)
        self.img_dict, self.mask_dict = {},{}
        for case in data_id_list:
            self.img_dict[case] = case+'.png'
            self.mask_dict[case] = case+'-mask.png'

    def __getitem__(self,idx):
        data_idx = self.data_id_list[idx]
        img = Image.open(os.path.join(self.data_dir, self.img_dict[data_idx])).resize((self.opt.img_size, self.opt.img_size),Image.LANCZOS).convert('RGB')
        labels = Image.open(os.path.join(self.data_dir, self.mask_dict[data_idx])).resize((self.opt.img_size,self.opt.img_size),Image.NEAREST).convert("L")

        if self.img_transforms is not None: 
            img = self.img_transforms(img)

        if self.transforms is not None:
            img, labels = MyRandomApply(img,labels,self.transforms, self.p)

        img = ttf.to_tensor(img)
        img = self.normalize(img)

        # Only one forground class
        labels = ttf.to_tensor(labels)>0.0001
        labels = labels.float()

        file_id = self.data_id_list[idx]
        return {'data': img.float(),
                'labels': labels,
                'file_id': file_id}



