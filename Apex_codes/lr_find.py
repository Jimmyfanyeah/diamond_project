# official modules
import torch
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
# self-defined modules
from utils.utils import init_DDP, build_model
from utils.data import get_dataloaders
from utils.loss import calculate_loss


def CSS_Proj(rank,world_size,opt):
    opt.rank = rank
    opt.world_size = world_size
    init_DDP(opt)
    if opt.train_or_test == 'train':

        print('Start ...') 
        train_dataloader, test_dataloader = get_dataloaders(opt)

        device = torch.device('cuda')
        model = build_model(opt).to(device)
        model = DDP(model,find_unused_parameters=True,broadcast_buffers=False)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)  

        from torch_lr_finder import LRFinder

        criterion = calculate_loss(opt)
        # optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=0)
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        lr_finder.range_test(train_dataloader, start_lr=1e-7, end_lr=10, num_iter=100, step_mode='exp')
        lr_finder.plot(log_lr=True) # to inspect the loss-learning rate graph
        lr_finder.reset() # to reset the model and optimizer to their initial state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diamond Project')
    # Phase
    parser.add_argument('--train_or_test',          type=str,           default='train',        help='train or test')
    # Data info
    parser.add_argument('--img_size',               type=int,           default=512,            help='Size of input image')
    parser.add_argument('--batch_size',             type=int,           default=64,            help='training batch size')
    parser.add_argument('--n_class',                type=int,           default=1,             help='number of classes')
    parser.add_argument('--augmentation',           type=int,           default=1,            help='with data augmentation?')
    # Train params
    parser.add_argument('--bce_weight',             type=float,         default=1,           help='weight of bce loss')
    # Some locations
    parser.add_argument('--data_path',              default='/media/hdd_4T/css_data/1class/clip128')
    parser.add_argument('--train_id_loc',           default='/media/hdd_4T/css_data/1class/txt128/id_train_clip.txt')
    parser.add_argument('--val_id_loc',             default='/media/hdd_4T/css_data/1class/txt128/id_val_clip.txt')
    # Random seed
    parser.add_argument('--manualSeed',             type=int,               default=0, help='manual seed')
    parser.add_argument('--opt_level',default='O1')#English letter O

    parser.add_argument('--gpu_number',             type=str,           default='0',              help='assign gpu')
    parser.add_argument('--num_workers',            type=int,           default=3,              help='num of workers in dataloader')
    opt = parser.parse_args()

    gpu_number = len(opt.gpu_number.split(','))
    mp.spawn(CSS_Proj,args=(gpu_number,opt),nprocs=gpu_number,join=True)