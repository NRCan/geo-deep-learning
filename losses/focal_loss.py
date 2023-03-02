# Source: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Simple pytorch implementation of focal loss introduced
    by *Lin et al.* (https://arxiv.org/pdf/1708.02002.pdf).
    """
    
    def __init__(self, gamma=2, alpha=0.5, weight=None, ignore_index=255):
        """Initialize the focal loss.

        Args:
            gamma (int, optional): exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Defaults to 2.
            alpha (float, optional): weighting factor in range (0,1) to balance positive vs negative examples. Defaults to 0.5.
            weight (Tensor, optional): a manual rescaling weight given to each class. Defaults to None.
            ignore_index (int, optional): target value that is ignored and does not contribute to the input gradient. Defaults to 255.
        """        
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        """Foward function use during trainning. 
       
        Args:
            preds (Tensor): the output from model.
            labels (Tensor): ground truth.

        Returns:
            Tensor: focal loss score.
        """        
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt

        return loss
