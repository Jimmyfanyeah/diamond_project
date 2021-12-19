import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.init import kaiming_normal_
# from bd_loss import BDLoss


def init_DDP(opt):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='123456'
    gpus = [g.strip() for g in opt.gpu_number.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES']=gpus[opt.rank]
    dist.init_process_group('NCCL',rank=opt.rank,world_size=opt.world_size)


def print_log(print_string, log):
    print("---> {}".format(print_string))
    log.write('---> {}\n'.format(print_string))
    log.flush()


def build_model(opt):
    # if opt.model_use == 'origin_UNet':
    from .UNet_Architecture import ResNetUNet
    model = ResNetUNet(n_class=opt.n_class)
    # elif opt.model_use == 'UNet':
    #     from .unet_V2 import UNet
    #     model = UNet(n_class=opt.n_class)
    #     model.apply(model_ini)

    return model


def print_metrics(metrics, epoch_samples, phase, log):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:.4f}".format(k, metrics[k] / epoch_samples))

    print_log("{}: {}".format(phase, ", ".join(outputs)),log)


def pause(info_str='Press <ENTER> to continue...'):
    input(info_str)




def pred_to_mask(pred):
    mask = pred > 0.5
    mask = mask.to(torch.float32)
    return mask

def loadAndFreeze(opt,device):
    pass



def model_ini(m):
    if type(m) == torch.nn.Conv2d:
        kaiming_normal_(m.weight.data,nonlinearity='relu')
        m.bias.data.fill_(0)


