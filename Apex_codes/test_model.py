# official modules
import os,math,itertools,torch
import numpy as np
from math import ceil
from PIL import Image
import csv
from skimage.segmentation import mark_boundaries as mkbdy
import torchvision.transforms.functional as ttf
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
# self-defined modules
from utils.data import TrainingDataset as dataset
from utils.loss import dice_loss
from shutil import copy2


def CutAndBatch(img,opt):
    step = opt.img_size
    _,H,W = img.shape
    h,w = opt.img_size, opt.img_size
    Nh, Nw = ceil(H/step), ceil(W/step)
    batchedImgs = torch.empty(Nh*Nw,3,h,w).to(img.device).fill_(0)
    for i,j in itertools.product(range(Nh),range(Nw)):
        start_h, end_h = i*step, i*step + h
        start_w, end_w = j*step, j*step + w
        if end_h > H:
            start_h, end_h = H-h, H
        if end_w > W:
            start_w, end_w = W-w, W

        batchedImgs[i*Nw+j,:] = img[:, start_h:end_h, start_w:end_w]

    return batchedImgs


def BatchAssemble(batchedImgs,ori_size,opt):
    step = opt.img_size
    _,H,W = ori_size
    h,w = opt.img_size, opt.img_size
    Nh, Nw = ceil(H/step), ceil(W/step)

    img = torch.empty([H,W]).to(batchedImgs.device).fill_(0)
    for i,j in itertools.product(range(Nh),range(Nw)):
        start_h, end_h = i*step, i*step + h
        start_w, end_w = j*step, j*step + w
        if end_h > H:
            start_h, end_h = H-h, H
        if end_w > W:
            start_w, end_w = W-w, W

        img[start_h:end_h, start_w:end_w] += batchedImgs[i*Nw+j,:]

    img = img>0
    return img


def visualize(img,label,pred,option='overlap'):
    img_np, label_np, pred_np = np.array(img), np.array(label), np.array(pred)

    if option == 'overlap':
        img_np_label = mkbdy(img_np,label_np,color=(0,1,0),outline_color=(0,0.9,0))
        img_np_double = mkbdy(img_np_label,pred_np,color=(0,0,1),outline_color=(0,0,0.9))
        return Image.fromarray(np.uint8(255*img_np_double))

    if option == 'separate':
        img_np_label = mkbdy(img_np,label_np,color=(0,1,0),outline_color=(0,0.9,0))
        img_np_pred = mkbdy(img_np,pred_np,color=(0,0,1),outline_color=(0,0,0.9))
        img_np_double = np.concatenate((img_np_label,img_np_pred),axis=1)
        return Image.fromarray(np.uint8(255*img_np_double))


def test_model(model,opt):
    model.eval()

    with open(opt.val_id_loc,'r') as pt:
        # cases = [f[:11] for f in pt]
        cases = [f.split('.')[0] for f in pt]
    print(cases)
    # Split cases for each rank
    cases.sort()
    l = math.ceil(len(cases)/opt.world_size)
    selfCase = cases[l*opt.rank:min(l*(opt.rank+1),len(cases))]
    testDataset = dataset(opt.data_path, selfCase, opt=opt)

    copy2(os.path.join(opt.data_path,'cut_locations.txt'),os.path.join(opt.save_path,'cut_locations.txt'))

    result = np.array(['diamond id', 'dice loss', 'bce loss', 'label sum'])

    with torch.no_grad():
        for i, (img, label, file_id) in enumerate(testDataset):
            img = img.to('cuda')
            label = label.to('cuda')

            # cut large image to small patches with size=opt.img_size
            batchedImgs = CutAndBatch(img,opt)  
            print(f'Original img {img.shape} -> batched imgs {batchedImgs.shape}')
            N = batchedImgs.shape[0]
            batchedPreds = torch.empty([N,1,opt.img_size,opt.img_size]).to('cuda').fill_(0)

            for ii in range(math.ceil(N/opt.batch_size)):
                tmpImgs = batchedImgs[ii*opt.batch_size:min((ii+1)*opt.batch_size,N),:]
                with autocast():
                    tmpPreds = model(tmpImgs)

                tmpPreds = torch.sigmoid(tmpPreds.float())>0.5
                tmpPreds = tmpPreds.float()
                batchedPreds[ii*opt.batch_size:min((ii+1)*opt.batch_size,N),:] = tmpPreds

            # assemble batchedPreds back to original image
            pred = BatchAssemble(batchedPreds.squeeze(),img.shape,opt)
            diceLoss = dice_loss(pred.unsqueeze(0).unsqueeze(0),label.unsqueeze(0)).cpu().item()
            bceLoss = F.binary_cross_entropy(pred.float().unsqueeze(0).unsqueeze(0),label.unsqueeze(0)).cpu().item()
            result = np.vstack((result,np.column_stack((file_id,diceLoss,bceLoss,int(label.sum().item())))))

            pred = ttf.to_pil_image(pred.float().cpu())
            pred.save(os.path.join(opt.save_path,file_id+'_pred.png'))

            img = Image.open(os.path.join(opt.data_path,file_id+'.png')).convert('RGB')
            pred_vis = visualize(img,ttf.to_pil_image(label.squeeze().cpu()),pred)
            pred_vis.save(os.path.join(opt.save_path,file_id+'_pred_vis.png'))
            if opt.rank==0:
                print('{}/{} finished!'.format(i+1,len(testDataset)))

    dist.barrier()
    if opt.rank == 0:
        # write the results to a csv file named "loc.csv" under the infer result save folder
        row_list = result.tolist()
        with open(os.path.join(opt.save_path,'test_loss.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)