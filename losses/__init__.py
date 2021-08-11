# All losses from: https://github.com/nyoki-mtl/pytorch-segmentation/tree/master/src/losses/multi

import torch.nn as nn

from .focal_loss import FocalLoss
from .lovasz_loss import LovaszSoftmax
from .ohem_loss import OhemCrossEntropy2d
from .focal_loss import FocalLoss
from .dice_loss import DiceLoss
from .boundary_loss import BoundaryLoss


class MultiClassCriterion(nn.Module):
    def __init__(self, loss_type='CrossEntropy', **kwargs):
        super().__init__()
        if loss_type == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss(**kwargs)  # FIXME: this error happens with CrossEntropy: https://discuss.pytorch.org/t/is-there-anybody-happen-this-error/17416
        elif loss_type == 'Lovasz':
            self.criterion = LovaszSoftmax(**kwargs)
        elif loss_type == 'OhemCrossEntropy':
            self.criterion = OhemCrossEntropy2d(**kwargs)
        elif loss_type == 'Focal':
            self.criterion = FocalLoss(**kwargs)
        elif loss_type == 'Dice':
            self.criterion = DiceLoss(**kwargs)
        elif loss_type == 'BF1':
            self.criterion = BoundaryLoss(**kwargs)
        elif loss_type == 'Duo':
            lst = [LovaszSoftmax(**kwargs), BoundaryLoss(**kwargs)]
            self.criterion = lst
        elif loss_type == 'bcewithlogitsloss':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError\
                (f'Current version of geo-deep-learning does not implement {loss_type} loss')

    def forward(self, preds, labels):
        # preds = F.interpolate(preds, labels.shape[1:], mode='bilinear', align_corners=True)
        if isinstance(self.criterion, list): #TODO: does list = MECnet_loss(output, o1, o2, o3, o4, o5)
            cals = []
            for obj in self.criterion:
                cals.append(obj(preds, labels))
            loss = sum(cals)
        else:
            loss = self.criterion(preds, labels)
        return loss
