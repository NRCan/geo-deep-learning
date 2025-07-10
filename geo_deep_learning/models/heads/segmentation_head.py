"""Segmentation head for semantic segmentation."""

from typing import NamedTuple

import torch
from torch import nn


class SegmentationOutput(NamedTuple):
    """Segmentation output."""

    out: torch.Tensor
    aux: torch.Tensor | None


class SegmentationHead(nn.Module):
    """Simple 1x1 convolution head for semantic segmentation."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        """Initialize segmentation head."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(x)
