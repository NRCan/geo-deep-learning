import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, eps=1e-7, ignore_index=None, weight=None):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        target = target.unsqueeze_(1)

        if torch.sum(target) == 0:
            output = 1.0 - output
            target = 1.0 - target

        return 1.0 - (2 * torch.sum(output * target) + self.smooth) / \
               (torch.sum(output) + torch.sum(target) + self.smooth + self.eps)