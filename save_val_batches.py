# Official modules
import argparse
import os
import torch.multiprocessing as mp
from shutil import copy2
# Self-defined modules
from Apex_codes.utils.data import get_dataloaders
from Apex_codes.utils.utils import init_DDP

def Save_val_batch(rank,world_size,opt):

    opt.rank = rank
    opt.world_size = world_size
    init_DDP(opt)

    save_path = '/media/hdd/diamond_data/UNet_seg_1class/val_batch_v3'

    train_dataloader, test_dataloader = get_dataloaders(opt)
    for ii, (input, label, sample_ids) in enumerate(test_dataloader):
        tmp_save_folder = os.path.join(save_path, f'rank_{opt.rank}',f'val_{ii}')
        os.makedirs(tmp_save_folder,exist_ok=True)
        for sample_id in sample_ids:
            copy2(os.path.join(opt.data_path,sample_id+'.png'),tmp_save_folder)
            copy2(os.path.join(opt.data_path,sample_id+'-mask.png'),tmp_save_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diamond Project')
    parser.add_argument('--img_size', type=int, default=64, help='Size of input image')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--n_class', type=int, default=1, help='number of classes')
    parser.add_argument('--augmentation', type=int, default=0, help='with data augmentation?')
    # Some locations
    parser.add_argument('--data_path', default="/media/hdd/diamond_data/UNet_seg_1class/clip256")
    parser.add_argument('--train_id_loc', default="/media/hdd/diamond_data/UNet_seg_1class/txt256/id_train_clip.txt")
    parser.add_argument('--val_id_loc', default="/media/hdd/diamond_data/UNet_seg_1class/txt256/id_val_clip.txt")
    parser.add_argument('--gpu_number', type=str, default='2,3', help='assign gpu')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers in dataloader')
    opt = parser.parse_args()


    gpu_number = len(opt.gpu_number.split(','))
    mp.spawn(Save_val_batch,args=(gpu_number,opt),nprocs=gpu_number,join=True)