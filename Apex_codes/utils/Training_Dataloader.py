# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:27:08 2019

@author: Who?
"""
from torch.utils.data import DataLoader
from .Training_Dataset import TrainingDataset,TrainingDataset_V2
from torch.utils.data.distributed import DistributedSampler as Disample

def dataloader(data_dir, data_id_list, batch_size=10, shuffle=False, num_workers=0,
               img_transforms=None, transforms=None,p=None,opt=None):
    
    dataset = TrainingDataset(data_dir, data_id_list, img_transforms, transforms, p, opt)
    return DataLoader(dataset, batch_size = batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)

def dataloader_V2(data_dir, data_id_list, batch_size=10, shuffle=False, num_workers=0,img_transforms=None, transforms=None,p=None,opt=None):
    dataset = TrainingDataset_V2(data_dir, data_id_list, img_transforms, transforms, p, opt)
    try:
        Sampler = Disample(dataset,num_replicas=opt.world_size,rank=opt.rank,shuffle=shuffle)
        dl = DataLoader(dataset,batch_size=batch_size,sampler=Sampler,num_workers=num_workers,pin_memory=True)
    except:
        dl = DataLoader(dataset,batch_size = batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    return dl

















