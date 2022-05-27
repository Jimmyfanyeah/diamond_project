import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
# from torch.nn.modules.module import _forward_unimplemented


def precision_loss(pred, target,smooth=1e-5,reduce=True):
    pred = pred.contiguous()
    target = target.contiguous()
    _,_,m,n = target.shape
    smooth = smooth*m*n

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = 1 - ((intersection + smooth) / (target.sum(dim=2).sum(dim=2) + smooth))

    if reduce:
        return loss.mean()
    else:
        return loss 


def dice_loss(pred, target,smooth=1e-5, reduce=True):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    if reduce:
        return loss.mean()
    else:
        return loss


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                    putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class calculate_loss(nn.Module):
    def __init__(self, opt):
        super(calculate_loss, self).__init__()
        self.bce_weight = opt.bce_weight
        self.n_class = opt.n_class
        # self.FocalLoss = FocalLoss(opt.n_class)

    def forward(self, pred, target, metric=None, metrics=None):

        bce = F.binary_cross_entropy_with_logits(pred,target,pos_weight=torch.ones(self.n_class,device=pred.device))

        pred = torch.sigmoid(pred)
        dice = dice_loss(pred, target)
        # focal_loss = self.FocalLoss(pred,target)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        # loss = focal_loss

        dice_dt,loss_dt = dice.detach().clone(),loss.detach().clone()
        bce_dt = bce.detach().clone()
        # focal_loss_dt = focal_loss.detach().clone()

        if metric is not None:
            metric['bce'] = bce_dt.float() 
            metric['dice'] = dice_dt.float()
            # metric['focal_loss'] = focal_loss_dt.float()
            metric['loss'] = loss_dt.float()

        if metrics is not None:
            metrics['bce'] += bce_dt.float() * target.size(0)
            metrics['dice'] += dice_dt.float() * target.size(0)
            # metrics['focal_loss'] += focal_loss_dt.float() * target.size(0)
            metrics['loss'] += loss_dt.float() * target.size(0)

        return loss




