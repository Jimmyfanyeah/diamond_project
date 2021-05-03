import time
import copy
from collections import defaultdict
import torch
from utils.utils import calc_loss, print_metrics, pause, print_log
import os
import datetime
from torch.cuda.amp import autocast,GradScaler
from utils.vat import VATLoss
import torch.distributed as dist
from math import inf

def train_model_V2(model, optimizer, scheduler, device, dataloaders, log, logger, opt, num_epochs=25):
    best_loss = inf
    best_train_loss = inf
    best_val_dice = inf
    not_improve = 0
    scaler = GradScaler()

    # if opt.resume_training:
    #     try:
    #         best_loss = float(torch.load(opt.model_load_dir)['epoch_loss'])
    #     except:
    #         pass

    for epoch in range(num_epochs):
        if opt.rank == 0:
            log.write('\n')
            print('\n')
            print_log('Epoch {}/{}'.format(epoch+1, num_epochs), log)
            print_log('-' * 10, log)
            print_log(time.ctime(time.time()), log)

        since = time.time()
        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                if opt.rank == 0:
                    for param_group in optimizer.param_groups:
                        print_log("lr"+ str(param_group['lr']), log)
            else:
                model.eval()

            metrics = defaultdict(float)
            metric = defaultdict(float)
            epoch_samples = 0

            for ii, sample in enumerate(dataloaders[phase]):
                inputs = sample['data'].to(device)
                labels = sample['labels'].to(device)
                sample_id = sample['file_id']
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward track history only in train
                with torch.set_grad_enabled(phase=='train'):

                    with autocast():
                        outputs = model(inputs)
                        loss = calc_loss(outputs,labels,metrics,metric,opt)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                epoch_samples += inputs.size(0)
                # print each iteration
                # if opt.rank==0:
                print('Epoch [{}/{}], {} {}/{}, bce: {:.4f}, dice:{:.4f}, \nfile ids: {}' \
                    .format(epoch+1,num_epochs,phase,ii+1,len(dataloaders[phase]),metric['bce'],metric['dice'], sample_id)) # optimizer.param_groups[0]['lr']

            # We allreduce losses recorded in metric. Note that after allreduce, losses in metrics contain the sum of all losses in different ranks.
            for key in metrics.keys():
                dist.all_reduce_multigpu([metrics[key]])

            if opt.rank == 0:
                print_metrics(metrics, epoch_samples*opt.world_size, phase,log)

            epoch_loss = metrics['loss']/epoch_samples/opt.world_size
            epoch_dice_loss = metrics['dice']/epoch_samples/opt.world_size

            if phase == 'train' and opt.lr_mode =='StepLR':
                scheduler.step()
            if phase == 'val' and opt.lr_mode == 'ReduceOnPlateau':
                scheduler.step(epoch_loss)

            # print loss in tensorboard
            if opt.rank==0:
                for key in metrics.keys():
                    logger.scalars_summary("All/{}".format(key), {phase:metrics[key]/epoch_samples/opt.world_size}, epoch+1)

            ## save checkpoints
            if phase == 'val' and epoch_loss < best_loss-1e-4:
                # deep copy the model
                if opt.rank == 0:
                    print_log('val loss improve from {:.4f} to {:.4f}, saving the best model....'.format(best_loss,epoch_loss), log)
                    os.makedirs(opt.model_save_dir,exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch_loss' : epoch_loss
                    }, os.path.join(opt.model_save_dir, opt.save_name + '-UNet_val_best'))
                not_improve = 0
                best_loss = epoch_loss
            elif phase == 'val':
                not_improve += 1
                if opt.rank==0:
                    print_log('Not improve in val loss for {} epochs, best val loss is {:.4f}'.format(not_improve, best_loss), log)


            if phase == 'val' and epoch_dice_loss < best_val_dice-1e-4:
                # deep copy the model
                if opt.rank == 0:
                    print_log('val dice loss improve from {:.4f} to {:.4f}, saving the best model....'.format(best_val_dice,epoch_dice_loss), log)
                    os.makedirs(opt.model_save_dir,exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch_loss' : epoch_loss
                    }, os.path.join(opt.model_save_dir, opt.save_name + '-UNet_val_dice_best'))
                best_val_dice = epoch_dice_loss
            elif phase == 'val':
                if opt.rank==0:
                    print_log('Not improve in val dice loss', log)


            if phase == 'train' and epoch_loss < best_train_loss-1e-4:
                # deep copy the model
                if opt.rank == 0:
                    print_log('train loss improve from {:.4f} to {:.4f}, saving the best model'.format(best_train_loss,epoch_loss), log)
                    os.makedirs(opt.model_save_dir,exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch_loss' : epoch_loss
                    }, os.path.join(opt.model_save_dir, opt.save_name + '-UNet_train_best'))
                best_train_loss = epoch_loss
            elif phase == 'train':
                # update stagnation indicator
                if opt.rank==0:
                    print_log('Not improve in train loss, best train loss is {:.4f}'.format(best_train_loss), log)


        time_elapsed = time.time() - since

        if opt.rank == 0:
            print_log('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), log)
            # save checkpoint
            if epoch%opt.saveEpoch == opt.saveEpoch-1:
                os.makedirs(opt.model_save_dir,exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_loss' : epoch_loss
                }, os.path.join(opt.model_save_dir, opt.save_name + '-UNet_checkpoint_'+str(epoch+1)))

        # if no improvement for more than ? epochs break training
        # if not_improve_train >= 30:
        #     break

    if opt.rank == 0:
        print_log('Best val loss: {:4f}'.format(best_loss), log)
        log.close()

    return model


def train_model_of(model, optimizer, scheduler, device, dataloaders, log, logger, opt, num_epochs=25):
    # save model with minimum loss on training set, not consider val set,---> overfitting case

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_loss = 1e10
    best_epoch = 1000
    not_improve = 0
    scaler = GradScaler()
    if opt.vat:
        vat = VATLoss(xi=5e-2,eps=2e-1)

    # if opt.resume_training:
    #     try:
    #         best_loss = float(torch.load(opt.model_load_dir)['epoch_loss'])
    #     except:
    #         pass

    for epoch in range(num_epochs):
        if opt.rank == 0:
            log.write('\n')
            print('\n')
            print_log('Epoch {}/{}'.format(epoch, num_epochs - 1), log)
            print_log('-' * 10, log)
            print_log(time.ctime(time.time()), log)

        since = time.time()
        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                if opt.rank == 0:
                    for param_group in optimizer.param_groups:
                        print_log("lr"+ str(param_group['lr']), log)

            else:
                model.eval()

            metrics = defaultdict(float)
            metric = defaultdict(float)
            epoch_samples = 0

            for ii, sample in enumerate(dataloaders[phase]):
                inputs = sample['data'].to(device)
                labels = sample['labels'].to(device)
                sample_id = sample['file_id']
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history only in train
                with torch.set_grad_enabled(phase=='train'):
                    if phase=='train' and opt.vat:
                        with autocast():
                            with torch.no_grad():
                                outputs = model(inputs)

                        with autocast():
                            vat_loss = vat(model,inputs,outputs,scaler)
                            outputs = model(inputs)
                            loss = calc_loss(outputs,labels,metrics,metric,opt)

                    else:
                        vat_loss = None
                        with autocast():
                            outputs = model(inputs)
                            loss = calc_loss(outputs,labels,metrics,metric,opt)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        # # compute gradient norm
                        if opt.rank==0:
                            total_norm = 0
                            for p in list(filter(lambda p: p.grad is not None, model.parameters())):
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                            total_norm = total_norm ** (1. / 2)

                epoch_samples += inputs.size(0)
                # print each iteration
                if opt.rank==0:
                    print('Epoch [{}/{}], {} {}/{}, Loss: {:.4f}, bce: {:.4f}, dice:{:.4f}, gradient: {:2f} \n sample ids: {}' \
                        .format(epoch+1,num_epochs,phase,ii+1,len(dataloaders[phase]),loss.item(),metric['bce'],metric['dice'],total_norm, sample_id)) # optimizer.param_groups[0]['lr']
                # print in tensorboard
                if opt.rank==0:
                    if phase == 'train':
                        logger.scalar_summary("Train/BCE", metric['bce'], ii+epoch*len(dataloaders[phase])+1)
                        logger.scalar_summary("Train/Dice", metric['dice'], ii+epoch*len(dataloaders[phase])+1)
                        logger.scalar_summary("Train/Loss", metric['loss'], ii+epoch*len(dataloaders[phase])+1)
                        if opt.vat:
                            logger.scalar_summary("Train/Vat", metric['vat'], ii+epoch*len(dataloaders[phase])+1)
                        
                        logger.scalar_summary('Other/Gradient norm',total_norm, ii+epoch*len(dataloaders[phase])+1)
                        logger.scalar_summary('Other/Learning rate',optimizer.param_groups[0]['lr'],ii+epoch*len(dataloaders[phase])+1)

                    elif phase == 'val':
                        logger.scalar_summary("Validation/BCE", metric['bce'], ii+epoch*len(dataloaders[phase])+1)
                        logger.scalar_summary("Validation/Dice", metric['dice'], ii+epoch*len(dataloaders[phase])+1)
                        logger.scalar_summary("Validation/Loss", metric['loss'], ii+epoch*len(dataloaders[phase])+1)
                        if opt.vat:
                            logger.scalar_summary("Validation/Vat", metric['vat'], ii+epoch*len(dataloaders[phase])+1)

            if phase == 'train' and opt.lr_mode =='StepLR':
                scheduler.step()

            # We allreduce losses recorded in metric. Note that after allreduce, losses in metrics contain the sum of all losses in different ranks.
            for key in metrics.keys():
                dist.all_reduce_multigpu([metrics[key]])

            if opt.rank == 0:
                print_metrics(metrics, epoch_samples*opt.world_size, phase,log)

            epoch_loss = metrics['loss']/epoch_samples/opt.world_size

            if phase == 'val' and opt.lr_mode == 'ReduceOnPlateau':
                scheduler.step(epoch_loss)

            if phase == 'train' and epoch_loss < best_loss:
                # deep copy the model
                if opt.rank == 0:
                    print_log('Train loss improve from {:6f} to {:6f}, saving the best model'.format(best_loss,epoch_loss), log)
                    os.makedirs(opt.model_save_dir,exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch_loss' : epoch_loss
                    }, os.path.join(opt.model_save_dir, opt.save_name + '-UNet_best'))
                not_improve = 0
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.module.state_dict())
            elif phase == 'train':
                # update stagnation indicator
                not_improve += 1
                if opt.rank==0:
                    print_log('No improvement in train loss for {} epochs, best train loss is {:6f} in epoch {}'.format(not_improve, best_loss, best_epoch), log)

        time_elapsed = time.time() - since

        if opt.rank == 0:
            print_log('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), log)
            # save checkpoint
            if epoch%opt.saveEpoch == opt.saveEpoch-1:
                os.makedirs(opt.model_save_dir,exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_loss' : epoch_loss
                }, os.path.join(opt.model_save_dir, opt.save_name + '-UNet_checkpoint_'+str(epoch+1)))

        # if no improvement for more than 20 epochs break training
        # if not_improve >= 40:
        #     break

    if opt.rank == 0:
        print_log('Best train loss: {:4f}'.format(best_loss), log)
        log.close()

    # load best model weights
    model.module.load_state_dict(best_model_wts)
    if opt.rank == 0:
        torch.save({
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch_loss' : best_loss
        }, os.path.join(opt.model_save_dir, opt.save_name + '-UNet_best'))

    return model

