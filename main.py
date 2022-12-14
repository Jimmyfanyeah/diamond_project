# Official modules
import argparse
import os
import random
from time import strftime
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# Self-defined modules
from efficientnet_pytorch import EfficientNet
from utils.logger import Logger
from utils.weighted_sampling import weighted_sampler_generator
from utils.dataset import DiamondDataset, DiamondDataset_zeropad

parser = argparse.ArgumentParser(description='PyTorch Diamond Training')
parser.add_argument('--data', metavar='/home/lingjia/Documents/diamond_data/inclusion_classification',  help='path to dataset')
parser.add_argument('--epochs', default=2, type=int, metavar='N',  help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',  help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N', 
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--image_size', default=224, type=int, help='image size')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_freq', default=10, type=int, help='checkpoint save frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--gpu', default='0,1', type=str, help='GPU id to use.')
parser.add_argument('--advprop', default=False, action='store_true', help='use advprop or not')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',  help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',  help='number of data loading workers (default: 4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--save_path', default=None, type=str, help='path for result. ')
parser.add_argument('--num_cls', default=None, type=int, help='number of classes ')
parser.add_argument('--img_cls', default=None, type=str)
parser.add_argument('--min_lr', default=None, type=float)
parser.add_argument('--max_lr', default=None, type=float)
parser.add_argument('--zeropad', default=0, type=int, help='whether to resize or zero-padding')

best_acc1 = 0

def main():
    args = parser.parse_args()
    args.img_cls = args.img_cls.split('-') if args.img_cls else None

    ngpus_per_node = len(args.gpu.split(','))
    args.distributed = ngpus_per_node>1   # args.world_size > 1 or args.multiprocessing_distributed

    # print args
    for k,v in args._get_kwargs():
        print('=> {}: {}'.format(k,v))

    if args.distributed:
        # # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        args.rank = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    args.world_size = ngpus_per_node
    os.makedirs(args.save_path,exist_ok=True)
    global best_acc1

    if args.rank == 0:
        logger = Logger(os.path.join(args.save_path, f"{strftime('%H-%M-%S')}_tb"))
        print("=> Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        init_DDP(args)

    # Create model
    if 'efficientnet' in args.arch:
        if args.pretrained:
            model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop)
            if args.rank == 0:
                print("=> using pre-trained model '{}'".format(args.arch))
        else:
            if args.rank == 0:
                print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(args.arch,num_classes=args.num_cls)
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()
            numFit = model.fc.in_features
            model.fc = nn.Linear(numFit, args.num_cls)

    model.cuda()
    if args.distributed:
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        # model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)


    # define loss function (criterion) and optimizer
    if args.num_cls == 1:
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr)
    # print(f'=> CyclicLR with min lr={args.min_lr} max lr={args.max_lr}')
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, threshold=0.0001, 
                                               min_lr=1e-7, eps=1e-08, verbose=True)
    # scheduler = None

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), f"=> No checkpoint found at {args.resume}"
        print(f"=> loading checkpoint {args.resume}")
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1'].to('cuda')
        # if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            # best_acc1 = best_acc1.to('cuda')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded epoch {checkpoint['epoch']})")


    cudnn.benchmark = True

    # Load dataset
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # if args.advprop:
    #     normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    # else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize(mean=[0.3364, 0.3477, 0.3473], std=[0.1371, 0.1415, 0.1424])

    if 'efficientnet' in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
    else:
        image_size = args.image_size
    if args.rank == 0:
        print('=> Using image size', image_size)

    transforms_list = [transforms.RandomRotation(30),
                       transforms.RandomVerticalFlip(0.5),
                       transforms.RandomHorizontalFlip(0.5)
                       ]

    if args.zeropad == 1:
        print('=> Create dataset use zero padding and center crop')
        train_dataset = DiamondDataset_zeropad(
            traindir,
            img_cls=args.img_cls,
            transform=transforms.Compose([
                transforms.RandomChoice(transforms_list),
                transforms.ToTensor(),
                normalize,])
        )
    elif args.zeropad == 0:
        print('=> Create dataset use resize')
        train_dataset = DiamondDataset(
            traindir,
            img_cls=args.img_cls,
            transform=transforms.Compose([
                transforms.Resize((image_size,image_size),interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomChoice(transforms_list),
                transforms.ToTensor(),
                normalize,])   
        )
    print(f"=> Classes: {train_dataset.classes}")
    print(f"=> Unique labels: {torch.unique(torch.tensor(train_dataset.targets), return_counts=True)}")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        # train_sampler = weighted_sampler_generator(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler)

    if args.zeropad == 1:
        print('=> Create dataset use zero padding and center crop')
        val_dataset = DiamondDataset_zeropad(
            valdir, 
            img_cls=args.img_cls, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,])
        )
    elif args.zeropad == 0:
        print('=> Create dataset use resize')
        val_dataset = DiamondDataset(
            valdir, 
            img_cls=args.img_cls, 
            transform=transforms.Compose([
                transforms.Resize((image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,])
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False) #pin_memory=True

    # Train
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        if args.rank == 0:
            print(f'\nEpoch {epoch}',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),f'LR:{get_lr(optimizer)}',)
        # train for one epoch
        acc1_train, acc2_train, loss_train = train(train_loader, model, criterion, optimizer, epoch, args, scheduler)

        if args.rank == 0:
            logger.scalars_summary("All/Acc@1", {'train':acc1_train}, epoch+1)
            # logger.scalars_summary("Acc@2", {'train':acc2_train}, epoch+1)
            logger.scalars_summary("All/loss", {'train':loss_train}, epoch+1)
            logger.scalar_summary("All/learning_rate", get_lr(optimizer), epoch+1)

        # evaluate on validation set
        acc1, acc2, loss_val = validate(val_loader, model, criterion, args)
        scheduler.step(loss_val)

        if args.rank==0:
            logger.scalars_summary("All/Acc@1", {'val':acc1}, epoch+1)
            # logger.scalars_summary("Acc@2", {'val':acc2}, epoch+1)
            logger.scalars_summary("All/loss", {'val':loss_val}, epoch+1)
    

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save latest checkpoint
        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(args.save_path,f'ckpt_latest.pth'))

        if is_best and args.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_path, epoch)

        if args.rank == 0 and epoch % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_path, epoch)

    if args.rank == 0 and args.distributed:
        dist.destroy_process_group()



def train(train_loader, model, criterion, optimizer, epoch, args, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@5', ':6.2f')
    if args.rank == 0:
        progress = ProgressMeter(len(train_loader), batch_time, losses, top1, top2, prefix="Train ")

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        images = images.to('cuda')
        target = target.to('cuda')

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(output, target, topk=(1,2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top2.update(acc2[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank==0:
            progress.print(i)
    
    return top1.avg, top2.avg, losses.avg



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    if args.rank == 0:
        progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top2, prefix='Test ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to('cuda')
            target = target.to('cuda')

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.rank==0:
                progress.print(i)

        if args.rank == 0:
            # TODO: this should also be done with the ProgressMeter
            print(f'=> TEST Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}')

    return top1.avg, top2.avg, losses.avg


def save_checkpoint(state, is_best, save_path, epoch):
    filename = f'ckpt_{epoch}.pth'
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init_DDP(opt):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='123456'
    gpus = [g.strip() for g in opt.gpu.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES']=gpus[opt.rank]
    dist.init_process_group('Gloo',rank=opt.rank,world_size=opt.world_size)



if __name__ == '__main__':
    main()
