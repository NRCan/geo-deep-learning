import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.getLogger(__name__)


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=19, theta=19, ignore_index=None, weight=None):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-batch
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        # logging.info(gt.shape)
        # zo = F.one_hot(gt, c)
        # logging.info(zo.shape)
        one_hot_gt = F.one_hot(gt, c).permute(0, 3, 1, 2).squeeze(dim=-1).contiguous().float()

        # boundary map
        gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        eps = 1e-7
        P = (torch.sum(pred_b * gt_b_ext, dim=2) + eps) / (torch.sum(pred_b, dim=2) + eps)
        R = (torch.sum(pred_b_ext * gt_b, dim=2) + eps) / (torch.sum(gt_b, dim=2) + eps)

        # Boundary F1 Score

        BF1 = (2 * P * R + eps) / (P + R + eps)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


# for debug
if __name__ == "__main__":
    import torch.optim as optim
    from torchvision.models import segmentation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = torch.randn(8, 3, 224, 224).to(device)
    gt = torch.randint(0, 10, (8, 224, 224)).to(device)

    logging.info(img.shape, gt.shape)

    model = segmentation.fcn_resnet50(num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = BoundaryLoss()

    y = model(img)

    loss = criterion(y['out'], gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)
