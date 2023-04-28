import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.getLogger(__name__)


def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label

    Args:
        label (_type_): _description_
        n_classes (_type_): _description_
        requires_grad (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: label on a form of an one hot vector.
    """
    label = label.squeeze(1).type(torch.long)
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad
    )[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


class BoundaryLoss(nn.Module):
    """
    Boundary Loss proposed in the paper *Boundary Loss for Remote Sensing Imagery Semantic Segmentation*
    from *Alexey Bokhovkin et al.* (https://arxiv.org/abs/1905.07852)
    
    From: https://github.com/yiskw713/boundary_loss_for_remote_sensing
    """

    def __init__(self, theta0=19, theta=19, ignore_index=None):
        """Initialize the boundary loss.

        Args:
            theta0 (int, optional): size of the sliding window. Defaults to 19.
            theta (int, optional): predened threshold on a distance. Defaults to 19.
            ignore_index (int, optional): index to be ignore during trainning. Defaults to None.
        """
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.ignore_index = ignore_index
        if self.ignore_index:
            logging.error(f'Ignore_index not implemented for Boundary Loss. Got ignore_index "{ignore_index}"')

    def forward(self, pred, gt):        
        """Foward function use during trainning. 

        Args:
            pred (Tensor): the output from model (before softmax), shape (N, C, H, W).
            gt (Tensor): ground truth, shape (N, H, W).

        Returns:
            Tensor: boundary loss score, averaged over mini-batch.
        """
        n, c, _, _ = pred.shape
        logging.debug(f"Prediction shape: {gt.shape}")

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        logging.debug(f"Ground truth shape: {gt.shape}")
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
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
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

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

    model = segmentation.fcn_resnet50(num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = BoundaryLoss()

    y = model(img)

    loss = criterion(y['out'], gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)