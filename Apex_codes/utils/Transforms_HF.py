# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:15:58 2019

@author: Who?
"""

import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

#scale = (0.9,1)
#ratio = (3./4.,4./3.)
#RandResizeCrop = T.RandomResizedCrop(512,scale,ratio)

def MyRandRotate(img, labels, angle=None):
    if angle is None:
        angle = random.uniform(0,359)
        
    img = TF.rotate(img, angle)
    try:
        labels = [TF.rotate(label, angle) for label in labels]
    except TypeError:
        labels = TF.rotate(labels, angle)
    return img, labels

def MyHFlip(img,labels):
    img = TF.hflip(img)
    try:
        labels = [TF.hflip(label) for label in labels]
    except TypeError:
        labels = TF.hflip(labels)
    return img, labels

def MyVFlip(img, labels):
    img = TF.vflip(img)
    try:
        labels = [TF.vflip(label) for label in labels]
    except TypeError:
        labels = TF.vflip(labels)
    return img, labels

#class MyRandGamma:
#    def __init__(self, r):
#        self.r = r
#        
#    def __call__(self, img):
#        gamma = random.uniform(1-self.r,1+self.r)
#        return ttf.functional.adjust_gamma(img, gamma)

class MyRandGamma:
    def __init__(self, r):
        self.r = r
        
    def __call__(self, img):
        gamma = random.uniform(1-self.r,1+self.r)
        return TF.adjust_gamma(img, gamma)

class MyRandResizedCrop:
    def __init__(self, scale, ratio):
        self.RandParams = T.RandomResizedCrop(512,scale,ratio)
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self,img,labels):
        params = self.RandParams.get_params(img,self.scale,self.ratio)
        img = TF.resized_crop(img,*params,img.size)
        try:
            labels = [TF.resized_crop(label,*params,img.size) for label in labels]
        except TypeError:
            labels = TF.resized_crop(labels, *params, img.size)
        return img, labels
    

def MyRandomApply(img,labels,transforms, p):
    transforms_tmp = random.sample(transforms, len(transforms))
    p_tmp = [p[transforms.index(x)] for x in transforms_tmp]
    for i,transform in enumerate(transforms_tmp):
        if random.random()<p_tmp[i]:
            img, labels = transform(img, labels)

    return img, labels


















