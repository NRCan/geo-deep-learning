import torch.nn as nn

class SegmentationHead(nn.Module):
    """
    Simple 1x1 convolution head for semantic segmentation.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the segmentation head.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)
        
        Returns:
            Tensor of shape (batch_size, num_classes, H, W) containing class logits
        """
        return self.conv(x)