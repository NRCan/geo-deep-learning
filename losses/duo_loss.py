from torch import nn

from losses.boundary_loss import BoundaryLoss
from losses.lovasz_loss import LovaszSoftmax


class DuoLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = [LovaszSoftmax(**kwargs), BoundaryLoss(**kwargs)]

    def forward(self, preds, labels):
        cals = []
        for obj in self.criterion:
            cals.append(obj(preds, labels))
        loss = sum(cals) / len(self.criterion)
        return loss
