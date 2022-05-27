# Official modules
import argparse
import os
from time import strftime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Self-defined modules
from utils.utils import init_DDP, print_log, build_model
from utils.logger import Logger
from utils.data import get_dataloaders
from train_model import train_model


def CSS_Proj(rank,world_size,opt):
    opt.rank = rank
    opt.world_size = world_size
    init_DDP(opt)
    if opt.train_or_test == 'train':
        if rank == 0:
            save_path = os.path.join(opt.save_path,opt.save_name)
            opt.save_path = save_path
            os.makedirs(save_path,exist_ok=True)
            log = open(os.path.join(save_path, 'log_{}.txt'.format(strftime('%H%M'))), 'w')
            # logger = None # tensorflow api conflicts bw e/l versions.
            logger = Logger(os.path.join(save_path, 'log_{}'.format(strftime('%m%d'))))
            print_log('save path : {}'.format(save_path), log)
            for k,v in opt._get_kwargs():
                print_log('{}: {}'.format(k,v),log)

        train_dataloader, test_dataloader = get_dataloaders(opt)
        dataloaders = {'train': train_dataloader, 'val': test_dataloader}

        device = torch.device('cuda')

        model = build_model(opt).to(device)
        model = DDP(model,find_unused_parameters=True,broadcast_buffers=False)

        optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)  

        if opt.resume:
            model.module.load_state_dict(torch.load(opt.model_path)['model'])
            # optimizer_ft.load_state_dict(torch.load(opt.model_path)['optimizer'])

        # opt.lr_mode = 'ReduceOnPlateau'
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=opt.lr_decay_per_epoch, gamma=opt.lr_decay_factor)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=opt.lr_decay_factor, patience=opt.lr_decay_per_epoch)

        if opt.rank == 0:
            print_log('Training ......', log)
            model = train_model(model, optimizer_ft, exp_lr_scheduler, device, dataloaders, log, logger, opt)
            dist.destroy_process_group()
            log.close()
        else:
            model = train_model(model, optimizer_ft, exp_lr_scheduler, device, dataloaders, None, None, opt)


    elif opt.train_or_test == 'test':
        if opt.rank == 0:
            print('\nTesting ......')
        model = build_model(opt).to('cuda')
        model = DDP(model,find_unused_parameters=True)
        model.module.load_state_dict(torch.load(opt.model_path)['model'])
        model.eval()
        from test_model import test_model
        if opt.rank==0:
            os.makedirs(opt.save_path,exist_ok=True)

        # dist.barrier()
        # test_model_clip(model,opt)
        test_model(model,opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diamond Project')
    # Phase
    parser.add_argument('--train_or_test', type=str, default='test', help='train or test')
    parser.add_argument('--resume', action='store_true', default=False)
    # Data info
    parser.add_argument('--img_size', type=int, default=64, help='Size of input image')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--n_class', type=int, default=1, help='number of classes')
    parser.add_argument('--augmentation', type=int, default=1, help='with data augmentation?')
    # Train params
    parser.add_argument('--Epoch', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.1, help='learning Rate')
    parser.add_argument('--lr_decay_per_epoch', type=int, default=30, help='number of epoches lr decay')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='lr decay factor')
    parser.add_argument('--bce_weight', type=float, default=0.5, help='weight of bce loss')
    parser.add_argument('--save_epoch', type=int, default=30, help='save model per number of epochs')
    # Some locations
    parser.add_argument('--data_path', default="")
    parser.add_argument('--train_id_loc', default="")
    parser.add_argument('--val_id_loc', default="")
    parser.add_argument('--save_path', default='', help='location to save results')
    # For validation
    parser.add_argument('--model_path', default='', help='location to load models')
    # Random seed
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
    parser.add_argument('--opt_level',default='O1')#English letter O
    parser.add_argument('--gpu_number', type=str, default='0', help='assign gpu')
    parser.add_argument('--num_workers', type=int, default=3, help='num of workers in dataloader')
    # parser.add_argument('--optimizer', type=str, default='adam', help='choose an optimizer')
    # parser.add_argument('--freeze_backbone', type=int, default=0, help='freeze backbone layer or not')
    opt = parser.parse_args()

    t = strftime('%Y%m%d') + \
                    '-imgSize' + str(opt.img_size) + \
                    '-bceWeight' + str(opt.bce_weight) + \
                    '-batchSize' + str(opt.batch_size) + \
                    '-Epoch' + str(opt.Epoch) + \
                    '-lr' + str(opt.lr)

    if opt.resume:
        t = t + '-resume'

    opt.save_name = t
    gpu_number = len(opt.gpu_number.split(','))
    mp.spawn(CSS_Proj,args=(gpu_number,opt),nprocs=gpu_number,join=True)

