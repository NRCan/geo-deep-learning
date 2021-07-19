import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(prb, lbl, ignore_index, only_present):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = prb.shape[0]
    prb = prb.permute(1, 2, 0).contiguous().view(-1, C)  # H * W, C
    lbl = lbl.view(-1)  # H * W
    if ignore_index is not None:
        valid_index = lbl != ignore_index
        if valid_index.sum() == 0:
            return torch.mean(prb * 0)
        prb = prb[valid_index]
        lbl = lbl[valid_index]

    total_loss = 0
    cnt = 0
    for c in range(C):
        fg = (lbl == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (fg - prb[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        total_loss += torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        cnt += 1
    try:
        return total_loss / cnt
    except Exception:
        logging.exception('Problem with lovasz_softmax_flat loss calculation')


class LovaszSoftmax(nn.Module):
    """
    Multi-class Lovasz-Softmax loss
      logits: [B, C, H, W] class logits at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      ignore_index: void class labels
    """

    def __init__(self, ignore_index=None, only_present=True, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = weight
        if weight is not None:
            logging.warning("The Lovasz function does not take weight parameter. "
                             "It inherently deals with class imbalance. See: https://arxiv.org/abs/1705.08790.")

    def forward(self, logits, labels):
        probas = F.softmax(logits, dim=1)
        total_loss = 0
        batch_size = logits.shape[0]
        for prb, lbl in zip(probas, labels):
            try:
                total_loss += lovasz_softmax_flat(prb, lbl, self.ignore_index, self.only_present)
            except TypeError as e:
                logging.exception('Problem with LovaszSoftmax loss calculation')
                try:
                    lbl_np = lbl.detach().cpu().numpy()
                    prb_np = prb.detach().cpu().numpy()
                    logging.debug(f'Lovasz loss cannot be calculated. '
                                  f'\n\tLabel unique values and count: {np.unique(lbl_np, return_counts=True)}'
                                  f'\n\tPrediction unique values and count: {np.unique(prb_np, return_counts=True)}')
                except:
                    logging.exception('Problem with logging label and prediction for debugging LovaszSoftmax')
        return total_loss / batch_size
