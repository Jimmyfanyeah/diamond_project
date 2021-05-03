import torch
import torch.nn.functional as F
import torch.distributed as dist
import os
from torch.nn.init import kaiming_normal_
# from bd_loss import BDLoss

def dice_loss(pred, target,smooth=1e-5,reduce=True):
    pred = pred.contiguous()
    target = target.contiguous()
    _,_,m,n = target.shape
    smooth = smooth*m*n

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    if reduce:
        return loss.mean()
    else:
        return loss


def calc_loss(pred, target, metrics, metric, opt):
    bce_weight = opt.bce_weight
    bce = F.binary_cross_entropy_with_logits(pred,target,pos_weight=50*torch.ones(opt.n_class,device=pred.device))

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)
    # loss = dice
    dice_dt,bce_dt,loss_dt = dice.detach().clone(),bce.detach().clone(),loss.detach().clone()
    metrics['bce'] += bce_dt.float() * target.size(0)
    metrics['dice'] += dice_dt.float() * target.size(0)
    metrics['loss'] += loss_dt.float() * target.size(0)

    metric['bce'] = bce_dt.float() 
    metric['dice'] = dice_dt.float()
    metric['loss'] = loss_dt.float()

    return loss

def print_metrics(metrics, epoch_samples, phase,log):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:.4f}".format(k, metrics[k] / epoch_samples))

    print_log("{}: {}".format(phase, ", ".join(outputs)),log)

def SplitData(with_target, test_cases=None):
    training = []
    validation = []

    if test_cases is not None:
        test_cases = [x.strip() for x in test_cases.split(',')]
    else:
        test_cases = list()

    for idx in with_target:
        case_id = idx[:11]
        if case_id in test_cases:
            validation.append(case_id)
        else:
            training.append(case_id)
    return training, validation

def SplitData_v2(with_target, test_cases=None):
    training = []
    validation = []

    if test_cases is not None:
        for l in test_cases:
            validation.append(l[:11])

    for idx in with_target:
        case_id = idx[:11]
        if not case_id in validation:
            training.append(case_id)
    return training, validation

def SplitData_v3(with_target, test_cases=None):
    training = []
    validation = []

    if test_cases is not None:
        for l in test_cases:
            validation.append(l[:-5])

    for idx in with_target:
        case_id = idx[:-5]
        if not case_id in validation:
            training.append(case_id)
    return training, validation

def pause(info_str='Press <ENTER> to continue...'):
    input(info_str)

def get_dataloaders_v1(opt):
    from .Training_Dataloader import dataloader
    # with_target_a = open(opt.data_dir[:-1]+ '/with_labels.txt\'','r')
    with_target_a = open('utils/with_labels.txt','r')
    training_id, validation_id = SplitData(with_target_a, opt.validation_cases)

    img_transforms, transforms, p = None, None, None
    if opt.augmentation:
        from .Transforms_HF import MyRandResizedCrop, MyHFlip, MyRandRotate, MyVFlip, MyRandGamma
        import torchvision.transforms as ttf
        scale = (0.95, 1)
        ratio = (5. / 6., 6. / 5.)
        RandResizedCrop = MyRandResizedCrop(scale, ratio)
        transforms = [MyHFlip, MyRandRotate, MyVFlip, RandResizedCrop]
        p = [0.5 for x in transforms]
        img_transforms = ttf.RandomApply([ttf.ColorJitter(0.2, 0.2, 0.2, 0.1), MyRandGamma(r=0.2)])


    train_dataloader = dataloader(os.path.join(opt.data_dir), training_id,
                                                      batch_size=opt.batchSize,
                                                      shuffle=True, num_workers=opt.num_workers,
                                                      img_transforms=img_transforms, transforms=transforms, p=p, opt=opt)

    test_dataloader = dataloader(os.path.join(opt.data_dir), validation_id,
                                                     batch_size=opt.batchSize,
                                                     shuffle=False, num_workers=opt.num_workers, opt=opt)
    return train_dataloader, test_dataloader

def get_dataloaders(opt):
    from .Training_Dataloader import dataloader
    with_target_a = open(opt.id_loc,'r')
    with_target_a_validation = open(opt.val_id_loc,'r')
    training_id, validation_id = SplitData_v3(with_target_a, with_target_a_validation)

    img_transforms, transforms, p = None, None, None
    if opt.augmentation and opt.train_or_test == "train":
        from .Transforms_HF import MyRandResizedCrop, MyHFlip, MyRandRotate, MyVFlip, MyRandGamma
        import torchvision.transforms as ttf
        scale = (0.95, 1)
        ratio = (5. / 6., 6. / 5.)
        RandResizedCrop = MyRandResizedCrop(scale, ratio)
        transforms = [MyHFlip, MyRandRotate, MyVFlip, RandResizedCrop]
        p = [0.5 for x in transforms]
        img_transforms = ttf.RandomApply([ttf.ColorJitter(0.2, 0.2, 0.2, 0.1), MyRandGamma(r=0.2)])

    train_dataloader = dataloader(os.path.join(opt.data_dir), training_id, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers, img_transforms=img_transforms, transforms=transforms, p=p, opt=opt)

    test_dataloader = dataloader(os.path.join(opt.data_dir), validation_id, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers, opt=opt)

    return train_dataloader, test_dataloader

def get_dataloaders_V2(opt):
    from .Training_Dataloader import dataloader_V2
    with_target_a = open(opt.id_loc,'r')
    with_target_a_validation = open(opt.val_id_loc,'r')
    training_id, validation_id = SplitData_v3(with_target_a, with_target_a_validation)
    training_id.sort()
    validation_id.sort()

    img_transforms, transforms, p = None, None, None
    if opt.augmentation and opt.train_or_test == "train":
        from .Transforms_HF import MyRandResizedCrop, MyHFlip, MyRandRotate, MyVFlip, MyRandGamma
        import torchvision.transforms as ttf
        scale = (0.95, 1)
        ratio = (5. / 6., 6. / 5.)
        RandResizedCrop = MyRandResizedCrop(scale, ratio)
        transforms = [MyHFlip, MyRandRotate, MyVFlip]
        p = [0.3 for x in transforms]
        img_transforms = ttf.RandomApply([ttf.ColorJitter(0.1, 0.1, 0.1, 0.1), MyRandGamma(r=0.1)])
        if opt.rank==0:
            print('Augmentation added!')


    train_dataloader = dataloader_V2(os.path.join(opt.data_dir), training_id, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers, img_transforms=img_transforms, transforms=transforms, p=p, opt=opt)

    test_dataloader = dataloader_V2(os.path.join(opt.data_dir), validation_id, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers, opt=opt)

    return train_dataloader, test_dataloader

def print_log(print_string, log):
  print("---> {}".format(print_string))
  log.write('---> {}\n'.format(print_string))
  log.flush()

def pred_to_mask(pred):
    mask = pred > 0.5
    mask = mask.to(torch.float32)
    return mask

def loadAndFreeze(opt,device):
    pass

def init_DDP(opt):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='29511'
    gpus = [g.strip() for g in opt.gpu_number.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES']=gpus[opt.rank]
    dist.init_process_group('NCCL',rank=opt.rank,world_size=opt.world_size)

def model_ini(m):
    if type(m) == torch.nn.Conv2d:
        kaiming_normal_(m.weight.data,nonlinearity='relu')
        m.bias.data.fill_(0)

def buildModel(opt):
    if opt.model_use == 'origin_UNet':
        from .UNet_Architecture import ResNetUNet
        model = ResNetUNet(n_class=opt.n_class)
    elif opt.model_use == 'UNet':
        from .unet_V2 import UNet
        model = UNet(n_class=opt.n_class)
        model.apply(model_ini)

    return model
