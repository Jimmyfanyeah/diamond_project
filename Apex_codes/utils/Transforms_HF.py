# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:15:58 2019
@author: Who?
"""

import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random


class MyRandGamma:
    # transformation for imgs
    def __init__(self, r):
        self.r = r
    def __call__(self, img):
        gamma = random.uniform(1-self.r,1+self.r)
        img = TF.adjust_gamma(img, gamma)
        return img


#scale = (0.9,1)
#ratio = (3./4.,4./3.)
#RandResizeCrop = T.RandomResizedCrop(512,scale,ratio)

def MyRandRotate(objects, angle=None):
    # objects = a list of img and labels
    if angle is None:
        angle = random.uniform(0,359)

    objects = [TF.rotate(obj, angle) for obj in objects]
    return objects


def MyHFlip(objects):
    # objects = a list of img and labels
    objects = [TF.hflip(obj) for obj in objects]
    return objects


def MyVFlip(objects):
    # objects = a list of img and labels
    objects = [TF.vflip(obj) for obj in objects]
    return objects


# class MyRandResizedCrop:
#     def __init__(self, scale, ratio):
#         self.RandParams = T.RandomResizedCrop(512,scale,ratio)
#         self.scale = scale
#         self.ratio = ratio

#     def __call__(self,img,labels):
#         params = self.RandParams.get_params(img,self.scale,self.ratio)
#         img = TF.resized_crop(img,*params,img.size)
#         try:
#             labels = [TF.resized_crop(label,*params,img.size) for label in labels]
#         except TypeError:
#             labels = TF.resized_crop(labels, *params, img.size)
#         return img, labels


class MyRandResizedCrop:
    def __init__(self, scale, ratio):
        self.RandResizedCrop = T.RandomResizedCrop(64,scale,ratio)
        self.scale = scale
        self.ratio = ratio

    def __call__(self,objects):
        objects = [self.RandResizedCrop(obj) for obj in objects]
        return objects


def MyRandomApply(objects,transforms, p):
    transforms_tmp = random.sample(transforms, len(transforms))
    p_tmp = [p[transforms.index(x)] for x in transforms_tmp]
    for i,transform in enumerate(transforms_tmp):
        if random.random()<p_tmp[i]:
            objects = transform(objects)

    return objects

















