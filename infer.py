import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import csv
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from math import floor
import torchvision.models as models
from collections import OrderedDict
import torch.nn as nn



def zeropad_func(image,image_size):
    w,h = image.size
    lp = (image_size[0]-w)/2
    rp = lp
    if not lp%1==0:
        # print(lp)
        lp = floor(lp)
        rp = lp + 1
    up = (image_size[1]-h)/2
    dp = up
    if not up%1==0:
        up = floor(up)
        dp = up + 1

    padding = transforms.Pad(padding=(int(lp),int(up),int(rp),int(dp)),fill=0,padding_mode='constant')
    image = padding(image)
    return image



if __name__ == '__main__':

    """ INPUT """
    phase = 'test'
    arch = 'resnet101'

    # data_path = f'/media/hdd/lingjia/hdd_diamond/cls/data/diamond/exp11/{phase}'
    data_path = f'/media/hdd/Bella/diamond_preprocessed/upscaled_diamond/{phase}'

    zeropad = 1 # 0 or 1
    save_path = '/media/hdd/lingjia/hdd_diamond/cls/result'
    os.makedirs(save_path,exist_ok=True)

    model_path = '/media/hdd/lingjia/hdd_diamond/cls/trained_model/2022-11-04-16-18-18_resnet101_resize_plain/model_best.pth'
    save_file = os.path.join(save_path,'2022-11-04-16-18-18_resnet101_resize_plain.csv')


    #######################
    """ Load test set """
    img_cls_str = None #'Twinning_wisp-other'
    img_cls = img_cls_str.split('-') if img_cls_str else None

    testList = {}
    class_list = {}
    data_list = os.listdir(data_path)
    data_list.sort()
    if img_cls is None:
        print('img_cls is None!')
        for idx, c in enumerate(data_list):
            class_list[c] = idx
    elif img_cls and 'other' in img_cls_str:
        print('1vsall case')
        idx = 0
        for c in data_list:
            if not c in img_cls:
                class_list[c] = len(img_cls)-1
            else:
                class_list[c] = idx
                idx += 1
    elif img_cls and not 'other' in img_cls_str:
        print('classes selected')
        for idx, c in enumerate(img_cls):
            class_list[c] = idx      
    print(class_list)
    num_cls = len(class_list) 

    for cls_name in class_list.keys():
        tmpList = os.listdir(os.path.join(data_path, cls_name))
        for tmp in tmpList:
            testList[tmp] = [tmp,class_list[cls_name],cls_name]

    # Load checkpoint
    if 'efficientnet' in arch:
        print("=> creating model '{}'".format(arch))
        model = EfficientNet.from_name(arch,num_classes=num_cls)
        image_size = EfficientNet.get_image_size(arch) # 224
    else:
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch]()
        numFit = model.fc.in_features
        model.fc = nn.Linear(numFit, num_cls)
        image_size = 224

    # checkpoint_path = os.path.join(model_path,'model_best.pth')
    checkpoint_path = model_path
    checkpoint = torch.load(checkpoint_path,map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # state_dict = torch.load(checkpoint_path)['state_dict']
    # # create new OrderedDict that does not contain 'module.'
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove 'module.'
    #     new_state_dict[name] = v
    # # load params
    # model.load_state_dict(new_state_dict)

    # infer_info = [['img_name','cls_name','cls_id',f'p_{img_cls[0]}',f'p_{img_cls[1]}']]
    infer_info = ['img_name','cls_name','cls_id']
    for key in class_list.keys():
        infer_info.append(f'p_{key}')
    print(infer_info)
    infer_info = [infer_info]
    num_idx = len(testList.keys())
    for idx, img_n in enumerate(testList.keys()):
        # Open image
        cls_name = testList[img_n][2]
        img = Image.open(os.path.join(data_path,cls_name,img_n))

        if zeropad == 1:
            img = zeropad_func(img, (image_size,image_size))
            tfms = transforms.Compose([
                                    # transforms.Resize((image_size,image_size)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    # transforms.Normalize(mean=[0.3364, 0.3477, 0.3473], std=[0.1371, 0.1415, 0.1424]),
                                    ])
        else:
            # Preprocess image
            tfms = transforms.Compose([
                                    transforms.Resize((image_size,image_size)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    # transforms.Normalize(mean=[0.3364, 0.3477, 0.3473], std=[0.1371, 0.1415, 0.1424]),
                                    ])
        
        img = tfms(img).unsqueeze(0)

        # Classify with EfficientNet
        model.eval()
        with torch.no_grad():
            logits = model(img)
 
        probs = torch.softmax(logits, dim=1)
        tmp_info = [img_n,cls_name,class_list[cls_name]]
        for pidx in range(probs.shape[1]):
            tmp_info.append(probs[0,pidx].item())
        print(f'{idx}/{num_idx} {img_n} GT:{cls_name} Pred:{probs}')

        infer_info.append(tmp_info)

    # infer_info_file = os.path.join(save_path,f'infer_on_{img_cls[0]}_{phase}.csv')
    infer_info_file = save_file
    with open(infer_info_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(infer_info)

