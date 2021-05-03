# official modules
import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import argparse
import os
import sys
from time import strftime
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as Disample
# self-defined modules
from utils.utils import get_dataloaders_V2, print_log, init_DDP,buildModel
from train_model import train_model_V2
from utils.logger import Logger

def CSS_Proj(rank,world_size,opt):
    opt.rank = rank
    opt.world_size = world_size
    init_DDP(opt)
    if opt.train_or_test == 'train':
        if rank == 0:
            results_vis_dir = os.path.join(opt.save_results_folder,opt.save_name)
            os.makedirs(results_vis_dir,exist_ok=True)
            log = open(os.path.join(results_vis_dir, 'log_{}.txt'.format(strftime('%H%M'))), 'w')
            # logger = None # tensorflow api conflicts bw e/l versions.
            logger = Logger(os.path.join(results_vis_dir, 'log_{}'.format(strftime('%m%d'))))
            print_log('save path : {}'.format(results_vis_dir), log)
            for k,v in opt._get_kwargs():
                print_log('{}: {}'.format(k,v),log)

        train_dataloader, test_dataloader = get_dataloaders_V2(opt)
        dataloaders = {
            'train': train_dataloader,
            'val': test_dataloader
        }
        device = torch.device('cuda')

        model = buildModel(opt).to(device)
        model = DDP(model,find_unused_parameters=True,broadcast_buffers=False)

        optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)  
        # optimizer freeze backbone layers
        if opt.freeze_backbone and opt.model_use == 'origin_UNet':
            for l in model.base_layers:
                for param in l.parameters():
                    param.requires_grad = False

        if opt.resume_training:
            model.module.load_state_dict(torch.load(opt.model_load_dir)['model'])
            # optimizer_ft.load_state_dict(torch.load(opt.model_load_dir)['optimizer'])

        opt.lr_mode = 'ReduceOnPlateau'
        if opt.lr_mode =='StepLR':
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=opt.lr_decay_per_epoch, gamma=opt.lr_decay_factor)
        elif opt.lr_mode == 'ReduceOnPlateau':
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=opt.lr_decay_factor, patience=opt.lr_decay_per_epoch, min_lr=5e-7)

        if opt.rank == 0:
            print_log('training ......', log)
            model = train_model_V2(model, optimizer_ft, exp_lr_scheduler, device, dataloaders, log, logger, opt, num_epochs=opt.Epoch)
            dist.destroy_process_group()
            log.close()
        else:
            model = train_model_V2(model, optimizer_ft, exp_lr_scheduler, device, dataloaders, None, None, opt, num_epochs=opt.Epoch)
    else:
        if opt.rank == 0:
            print('testing')
        model = buildModel(opt).to('cuda')
        model = DDP(model,find_unused_parameters=True)
        model.module.load_state_dict(torch.load(opt.model_load_dir)['model'])
        model.eval()
        from testModel import testModel
        if opt.rank==0:
            os.makedirs(opt.save_images_folder,exist_ok=True)

        dist.barrier()
        # testModel_clip(model,opt)
        testModel(model,opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diamond Project')
    parser.add_argument('--img_size',               type=int,           default=512,            help='Size of input image')
    parser.add_argument('--batchSize',              type=int,           default=7,              help='training batch size')
    parser.add_argument('--Epoch',                  type=int,           default=150,           help='number of epochs to train for')
    parser.add_argument('--num_workers',            type=int,           default=3,              help='num of workers in dataloader')
    parser.add_argument('--lr',                     type=float,         default=5e-5,           help='learning Rate')
    parser.add_argument('--optimizer',              type=str,           default='adam',         help='choose an optimizer')
    parser.add_argument('--lr_decay_per_epoch',     type=int,           default=30,            help='number of epoches lr decay')
    parser.add_argument('--lr_decay_factor',        type=float,         default=0.5,            help='lr decay factor')
    parser.add_argument('--freeze_backbone',        type=int,           default=0,              help='freeze backbone layer or not')
    parser.add_argument('--train_or_test',          type=str,           default='train',        help='train or test')
    parser.add_argument('--n_class',                type=int,           default=1,              help='number of classes')
    parser.add_argument('--model_use',              type=str,           default='origin_UNet',         help='origin_UNet or UNet')
    parser.add_argument('--augmentation',           type=int,           default= 1,            help='with data augmentation?')
    parser.add_argument('--bce_weight',             type=float,         default=0.5,            help='weight of bce loss')
    parser.add_argument('--gpu_number',             type=str,           default='0',              help='assign gpu')
    parser.add_argument('--saveEpoch',              type=int,           default=30,            help='save model per number of epochs')
    parser.add_argument('--resume_training',                 type=int,           default=0,             help='whether to resume train')
    # Some locations
    parser.add_argument('--data_dir',               default="/home/lingjia/Documents/CSS_Project_1/hdr_diamonds_labels_clipped")
    parser.add_argument('--id_loc',                 default="/home/lingjia/Documents/CSS_Project_1/hdr_diamonds_labels_clipped/allImgClipped.txt")
    parser.add_argument('--val_id_loc',             default="/home/lingjia/Documents/CSS_Project_1/hdr_diamonds_labels_clipped/id_validation.txt")
    parser.add_argument('--model_save_dir',         default='/home/lingjia/Documents/chow-unet-1_models',     help='location to save models')
    parser.add_argument('--save_results_folder',    default='/home/lingjia/Documents/chow-unet-1_results',    help='location to save results')
    # for validation
    parser.add_argument('--model_load_dir',         default='aaa',     help='location to save models')
    parser.add_argument('--save_images_folder',     default='/home/ljdai2/Documents/chow/Apex_main_V3_1_class_imgs/tmp',     help='resulting images')
    # random seed
    parser.add_argument('--manualSeed',             type=int,               default=0, help='manual seed')
    parser.add_argument('--opt_level',default='O1')#English letter O
    opt = parser.parse_args()
    t = strftime('%Y%m%d') + \
                        "-batchSize_" + str(opt.batchSize) + \
                        "-Epoch_" + str(opt.Epoch) + \
                        "-lr_" + str(opt.lr) + \
                        '-re_' + str(opt.resume_training)

    opt.save_name = t
    gpu_number = len(opt.gpu_number.split(','))
    mp.spawn(CSS_Proj,args=(gpu_number,opt),nprocs=gpu_number,join=True)