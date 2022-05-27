import os
import time
import copy
from math import inf,floor
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast,GradScaler
# self-defined modules
from utils.utils import print_metrics, print_log
from utils.loss import calculate_loss


def train_model(model, optimizer, scheduler, device, dataloaders, log, logger, opt):
    num_epochs = opt.Epoch
    best_loss, best_train_loss, best_val_dice = inf, inf, inf
    not_improve = 0
    dice_not_improve = 0
    scaler = GradScaler()
    calc_loss = calculate_loss(opt)

    train_start_time = time.time() # starting time of training
    for epoch in range(num_epochs):
        if opt.rank == 0:
            log.write('\n')
            print('\n')
            print_log('Epoch {}/{}'.format(epoch+1, num_epochs), log)
            print_log('-' * 10, log)
            print_log(time.ctime(time.time()), log)
            print_log('lr '+str(optimizer.param_groups[0]['lr']),log)

        epoch_start_time = time.time()
        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            metrics, metric = defaultdict(float), defaultdict(float)
            epoch_samples = 0

            for ii, (inputs, labels, sample_id) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward track history only in train
                with torch.set_grad_enabled(phase=='train'):

                    with autocast():
                        outputs = model(inputs)
                        loss = calc_loss(outputs,labels,metric,metrics)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                epoch_samples += inputs.size(0)
                # print each iteration
                if opt.rank==0:
                    print('Epoch {}/{}, {} {}/{}, bce:{:.4f}, dice:{:.4f}' \
                        .format(epoch+1,num_epochs,phase,ii+1,len(dataloaders[phase]),metric['bce'],metric['dice'])) 

            # We allreduce losses recorded in metric. Note that after allreduce, losses in metrics contain the sum of all losses in different ranks.
            for key in metrics.keys():
                dist.all_reduce_multigpu([metrics[key]])

            if opt.rank == 0:
                print_metrics(metrics, epoch_samples*opt.world_size, phase,log)

            epoch_loss = metrics['loss']/epoch_samples/opt.world_size
            epoch_dice_loss = metrics['dice']/epoch_samples/opt.world_size

            # if phase == 'train' and opt.lr_mode =='StepLR':
            #     scheduler.step()
            if phase == 'val':
                scheduler.step(epoch_loss)

            # print loss in tensorboard
            if opt.rank==0:
                for key in metrics.keys():
                    logger.scalars_summary("All/{}".format(key), {phase:metrics[key]/epoch_samples/opt.world_size}, epoch+1)

            ## save checkpoints
            if phase == 'val' and epoch_loss < best_loss-1e-4:
                if opt.rank == 0:
                    print_log('Val loss improve from {:.4f} to {:.4f}, save best model...'.format(best_loss,epoch_loss), log)
                    os.makedirs(opt.save_path,exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch_loss' : epoch_loss
                    }, os.path.join(opt.save_path, opt.save_name + '-UNet_val_best'))
                not_improve = 0
                best_loss = epoch_loss
            elif phase == 'val':
                not_improve += 1
                if opt.rank==0:
                    print_log('Val loss not improve for {} epochs, best val loss: {:.4f}'.format(not_improve, best_loss), log)

            if phase == 'val' and epoch_dice_loss < best_val_dice-1e-4:
                # deep copy the model
                if opt.rank == 0:
                    print_log('Val dice improve from {:.4f} to {:.4f}, save the best model...'.format(best_val_dice,epoch_dice_loss), log)
                    os.makedirs(opt.save_path,exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch_loss' : epoch_loss
                    }, os.path.join(opt.save_path, opt.save_name + '-UNet_val_dice_best'))
                dice_not_improve = 0
                best_val_dice = epoch_dice_loss
            elif phase == 'val':
                dice_not_improve += 1
                if opt.rank==0:
                    print_log('Val dice not improve for {} epochs, best val loss: {:.4f}'.format(dice_not_improve, best_val_dice), log)


            if phase == 'train' and epoch_loss < best_train_loss-1e-4:
                # deep copy the model
                if opt.rank == 0:
                    print_log('Train loss improve from {:.4f} to {:.4f}, save best model...'.format(best_train_loss,epoch_loss), log)
                    os.makedirs(opt.save_path,exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch_loss' : epoch_loss
                    }, os.path.join(opt.save_path, opt.save_name + '-UNet_train_best'))
                best_train_loss = epoch_loss
            elif phase == 'train':
                # update stagnation indicator
                if opt.rank==0:
                    print_log('Train loss not improve, best train loss: {:.4f}'.format(best_train_loss), log)

        time_elapsed = time.time() - epoch_start_time

        if opt.rank == 0:
            print_log('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), log)
            # save checkpoint
            os.makedirs(opt.save_path,exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_loss' : epoch_loss
            }, os.path.join(opt.save_path, opt.save_name + '-UNet_checkpoint_latest'))

            if epoch%opt.save_epoch == opt.save_epoch-1:
                os.makedirs(opt.save_path,exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_loss' : epoch_loss
                }, os.path.join(opt.save_path, opt.save_name + '-UNet_checkpoint_'+str(epoch+1)))

        # if no improvement for more than ? epochs break training
        if min(not_improve, dice_not_improve)>= 20 or optimizer.param_groups[0]['lr']<1e-7:
            break

    if opt.rank == 0:
        print_log('Best val loss: {:4f}'.format(best_loss), log)
        # print time
        time_elapsed_entire = time.time() - train_start_time
        print_log('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed_entire // 3600, 
            floor((time_elapsed_entire/3600 - time_elapsed_entire // 3600)*60), time_elapsed_entire % 60), log)

        log.close()

    return model


