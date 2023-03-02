from torch import nn

from losses.boundary_loss import BoundaryLoss
from losses.lovasz_loss import LovaszSoftmax


class DuoLoss(nn.Module):
    """
    Implementation of a losses combinaison between 
    the lovasz loss and the boundary loss.
    """
    def __init__(self, **kwargs):
        """Initialize the two losses.
        """        
        super().__init__()
        self.criterion = [LovaszSoftmax(**kwargs), BoundaryLoss(**kwargs)]

    def forward(self, preds, labels):
        """Foward function use during trainning. 
        
        Args:
            preds (Tensor): the output from model.
            labels (Tensor): ground truth.

        Returns:
            Tensor: duo loss score.
        """        
        cals = []
        for obj in self.criterion:
            cals.append(obj(preds, labels))
        loss = sum(cals) / len(self.criterion)
        return loss
