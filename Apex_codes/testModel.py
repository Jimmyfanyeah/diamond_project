from torch.cuda.amp import autocast
import torchvision.transforms.functional as ttf
import torch.nn.functional as F
import os,math,itertools,torch,datetime,time
from PIL import Image
from utils.utils import pred_to_mask
from utils.Training_Dataset import TrainingDataset_V2 as dataset
from utils.Training_Dataloader import dataloader_V2
from utils.utils import dice_loss, precision_loss
import torch.distributed as dist
import numpy as np
from skimage.segmentation import mark_boundaries as mkbdy
import csv

def cutAndBatch(img):
    step = 256
    batchedImgs = torch.empty(25,3,512,512).to(img.device).fill_(0)
    for i,j in itertools.product(range(5),range(5)):
        batchedImgs[i*5+j,:] = img[:,i*step:i*step+512,j*step:j*step+512]
    
    return batchedImgs

def assemble(imgs):
    img = torch.empty([1536,1536]).to(imgs.device).fill_(0)
    step = 256
    for i,j in itertools.product(range(5),range(5)):
        img[i*step:i*step+512,j*step:j*step+512] += imgs[i*5+j,:]

    return img

def drawAndPatch(img,label,pred):
    img_np,label_np,pred_np = np.array(img),np.array(label),np.array(pred)
    img_np_label = mkbdy(img_np,label_np,color=(0,1,0),outline_color=(0,0.9,0))
    img_np_pred = mkbdy(img_np,pred_np,color=(0,0,1),outline_color=(0,0,0.9))
    img_double = np.concatenate((img_np_label,img_np_pred),axis=1)
    return Image.fromarray(np.uint8(255*img_double))

def drawAndPatch_v2(img,label,pred):
    # overlap
    img_np,label_np,pred_np = np.array(img),np.array(label),np.array(pred)
    img_np_label = mkbdy(img_np,label_np,color=(0,1,0),outline_color=(0,0.9,0))
    img_np_pred = mkbdy(img_np_label,pred_np,color=(0,0,1),outline_color=(0,0,0.9))
    img_double = img_np_pred
    # img_double = img_np_label
    return Image.fromarray(np.uint8(255*img_double))


def testModel(model,opt):
    model.eval()
    opt.img_size = 1536 # 1200/400*512
    with open(opt.id_loc,'r') as pt:
        cases = [f.split('.')[0] for f in pt]

    pt = open(os.path.join(opt.save_images_folder,'dice_losses_{}.txt'.format(opt.rank)),'w')
    # Split cases for each rank.
    cases.sort()
    l = math.ceil(len(cases)/opt.world_size)
    selfCase = cases[l*opt.rank:min(l*(opt.rank+1),len(cases))]
    testDataset = dataset(opt.data_dir,selfCase,opt=opt)
    with torch.no_grad():
        for i,sample in enumerate(testDataset):
            img,label,file_id = sample['data'].to('cuda'),sample['labels'].to('cuda'),sample['file_id']
            batchedImgs = cutAndBatch(img)
            batchedPreds = torch.empty([25,1,512,512]).to('cuda').fill_(0)
            for ii in range(math.ceil(25/opt.batchSize)):
                tmpImgs = batchedImgs[ii*opt.batchSize:min((ii+1)*opt.batchSize,25),:]
                with autocast():
                    pred = model(tmpImgs)

                pred = torch.sigmoid(pred.float())>0.5
                pred = pred.float()
                batchedPreds[ii*opt.batchSize:min((ii+1)*opt.batchSize,25),:] = pred

            pred = assemble(batchedPreds.squeeze())>0.5
            diceLoss = dice_loss(pred.unsqueeze(0).unsqueeze(0),label.unsqueeze(0)).cpu().item()
            precisionLoss = precision_loss(pred.unsqueeze(0).unsqueeze(0),label.unsqueeze(0)).cpu().item()
            bceLoss = F.binary_cross_entropy(pred.float().unsqueeze(0).unsqueeze(0),label.unsqueeze(0)).cpu().item()

            pred = ttf.to_pil_image(pred.float().cpu())
            pred.resize((1200,1200),Image.NEAREST).save(os.path.join(opt.save_images_folder,file_id+'-pred.png'))
            pt.write('{},{:.4f},{:.4f},{:.4f},{}\n'.format(file_id,diceLoss,bceLoss,precisionLoss,int(label.sum().item())))
            img = Image.open(os.path.join(opt.data_dir,file_id+'.png')).convert('RGB')
            img = img.resize((1536,1536),Image.LANCZOS)
            img = drawAndPatch_v2(img,ttf.to_pil_image(label.squeeze().cpu()),pred)
            img.save(os.path.join(opt.save_images_folder,file_id+'_double.png'))
            if opt.rank==0:
                print('{}/{} finished!'.format(i+1,len(testDataset)))

    pt.close()
    dist.barrier()
    if opt.rank == 0:
        pt = open(os.path.join(opt.save_images_folder,'dice_losses.txt'),'w')
        for i in range(opt.world_size):
            with open(os.path.join(opt.save_images_folder,'dice_losses_{}.txt'.format(i)),'r') as tmpPt:
                pt.writelines(tmpPt.readlines())

        pt.close()


def testModel_clip(model,opt):
    # test on small patches with size 512-by-512
    model.eval()
    opt.img_size = 512
    with open(opt.id_loc,'r') as pt:
        cases = [f.split('.')[0] for f in pt]

    pt = open(os.path.join(opt.save_images_folder,'dice_losses_{}.txt'.format(opt.rank)),'w')

    # Split cases for each rank.
    cases.sort()
    l = math.ceil(len(cases)/opt.world_size)
    selfCase = cases[l*opt.rank:min(l*(opt.rank+1),len(cases))]
    testDataset = dataset(opt.data_dir,selfCase,opt=opt)
    with torch.no_grad():
        for i,sample in enumerate(testDataset):
            img,label,file_id = sample['data'].to('cuda'),sample['labels'].to('cuda'),sample['file_id']
            with autocast():
                pred = model(img.unsqueeze(0))
            # pred_np = np.array(pred.cpu())
            # np.save(os.path.join(opt.save_images_folder,"pred_{}.npy".format(file_id)),pred_np)
            # label_np = np.array(label.unsqueeze(0).cpu())
            # np.save(os.path.join(opt.save_images_folder,"label_{}.npy".format(file_id)),label_np)

            diceLoss = dice_loss(torch.sigmoid(pred.float())>0.5,label.unsqueeze(0))
            precisionLoss = precision_loss(torch.sigmoid(pred.float()),label.unsqueeze(0))
            bceLoss = F.binary_cross_entropy_with_logits(pred,label.unsqueeze(0))
            bceLoss_v1 = F.binary_cross_entropy_with_logits(pred,label.unsqueeze(0),pos_weight=100*torch.ones(opt.n_class,device=pred.device))

            pred = torch.sigmoid(pred.float())>0.5
            pred = pred.float()
            # pred_np = np.array(pred.cpu())
            # np.save(os.path.join(opt.save_images_folder,"pred_{}_v1.npy".format(file_id)),pred_np)
            pred = ttf.to_pil_image(pred.squeeze().float().cpu())
            pt.write('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {}\n'.format(file_id, bceLoss, bceLoss_v1, diceLoss, precisionLoss, int(label.sum().item())))
            img = Image.open(os.path.join(opt.data_dir,file_id+'.png'))
            img = img.resize((512,512),Image.LANCZOS)
            img = drawAndPatch(img,ttf.to_pil_image(label.squeeze().cpu()),pred)
            img.save(os.path.join(opt.save_images_folder,file_id+'_double.png'))
            if opt.rank==0:
                print('{}/{} finished!'.format(i+1,len(testDataset)))

    pt.close()
    dist.barrier()
    if opt.rank == 0:
        pt = open(os.path.join(opt.save_images_folder,'dice_losses.txt'),'w')
        for i in range(opt.world_size):
            with open(os.path.join(opt.save_images_folder,'dice_losses_{}.txt'.format(i)),'r') as tmpPt:
                pt.writelines(tmpPt.readlines())
        pt.close()


