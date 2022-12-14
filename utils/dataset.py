import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
from math import floor
import torchvision.transforms as transforms

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class DiamondDataset_zeropad(Dataset):
    def __init__(self, img_dir, img_cls=None, transform=None, tar_size=(224,224)):
        self.img_dir = img_dir
        self.transform = transform
        self.tar_size = tar_size

        # select classes from folder
        if img_cls:
            img_cls.sort()
        if img_cls and not 'other' in img_cls:
            img_cls.sort()
            self.img_folders = [n for n in os.listdir(self.img_dir) if n in img_cls]
        else:
            self.img_folders = os.listdir(self.img_dir)
        self.img_folders.sort()
        
        if img_cls:
            self.classes = img_cls
        else:
            self.classes = self.img_folders

        # generate label index for each class
        cls_idx = {}
        idx = 0
        for c in self.img_folders:
            if c in self.classes:
                cls_idx[c] = idx
                idx += 1
            else:
                cls_idx[c] = len(self.classes)-1
        
        # generate image dict
        self.img_list = []
        self.img_label = {}
        self.img_label_idx = {}
        for f in self.img_folders:
            imgs = os.listdir(os.path.join(self.img_dir,f))
            for img in imgs:
                if img in self.img_list:
                    print(f'{f} {img} exist in {self.img_label[img]}')
                    continue
                self.img_list.append(img)
                self.img_label[img] = f
                self.img_label_idx[img] = cls_idx[f]

        self.targets = list(self.img_label_idx.values())
                
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]

        img_path = os.path.join(self.img_dir, self.img_label[img], img)
        # image = read_image(img_path)
        image = Image.open(img_path)

        label = self.img_label_idx[img]

        # # zero-padding
        # image = Image.open(img_path)
        # w,h = image.size
        # lp = floor((tar_size[0]-w)/2)
        # up = floor((tar_size[1]-h)/2)
        # p = Image.new('RGB', tar_size, (0, 0, 0))
        # p.paste(image, (lp, up, w+lp, h+up))

        w,h = image.size
        lp = (self.tar_size[0]-w)/2
        rp = lp
        if not lp%1==0:
            # print(lp)
            lp = floor(lp)
            rp = lp + 1
        up = (self.tar_size[1]-h)/2
        dp = up
        if not up%1==0:
            up = floor(up)
            dp = up + 1

        padding = transforms.Pad(padding=(int(lp),int(up),int(rp),int(dp)),fill=0,padding_mode='constant')
        image = padding(image)

        if self.transform:
            image = self.transform(image)

        return image, label


class DiamondDataset(Dataset):
    def __init__(self, img_dir, img_cls=None, transform=None, tar_size=(224,224)):
        self.img_dir = img_dir
        self.transform = transform
        self.tar_size = tar_size

        # select classes from folder
        if img_cls:
            img_cls.sort()
        if img_cls and not 'other' in img_cls:
            img_cls.sort()
            self.img_folders = [n for n in os.listdir(self.img_dir) if n in img_cls]
        else:
            self.img_folders = os.listdir(self.img_dir)
        self.img_folders.sort()
        
        if img_cls:
            self.classes = img_cls
        else:
            self.classes = self.img_folders

        # generate label index for each class
        cls_idx = {}
        idx = 0
        for c in self.img_folders:
            if c in self.classes:
                cls_idx[c] = idx
                idx += 1
            else:
                cls_idx[c] = len(self.classes)-1
        
        # generate image dict
        self.img_list = []
        self.img_label = {}
        self.img_label_idx = {}
        for f in self.img_folders:
            imgs = os.listdir(os.path.join(self.img_dir,f))
            for img in imgs:
                if img in self.img_list:
                    print(f'{f} {img} exist in {self.img_label[img]}')
                    continue
                self.img_list.append(img)
                self.img_label[img] = f
                self.img_label_idx[img] = cls_idx[f]

        self.targets = list(self.img_label_idx.values())
                
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]

        img_path = os.path.join(self.img_dir, self.img_label[img], img)
        image = Image.open(img_path)

        label = self.img_label_idx[img]

        if self.transform:
            image = self.transform(image)

        return image, label