"""FCN head for semantic segmentation."""

import torch
from torch import nn

from geo_deep_learning.models.utils import ConvModule


class FCNHead(nn.Module):
    """
    FCN head for semantic segmentation.

    Adapted from https://github.com/open-mmlab/mmsegmentation

    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        channels: int = 256,
        in_index: int = -1,
        num_convs: int = 2,
        num_classes: int = 19,
        dropout_ratio: float = 0.1,
        *,
        concat_input: bool = False,
    ) -> None:
        """Initialize FCN head."""
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.in_index = in_index
        self.num_classes = num_classes
        self.concat_input = concat_input

        convs = []
        convs.append(
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                inplace=True,
            ),
        )

        convs.extend(
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                inplace=True,
            )
            for _ in range(num_convs - 1)
        )

        self.convs = nn.Identity() if num_convs == 0 else nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = ConvModule(
                in_channels + channels,
                channels,
                kernel_size=3,
                padding=1,
                inplace=True,
            )

        self.dropout = (
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        )

        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        x = inputs[self.in_index] if isinstance(inputs, (list, tuple)) else inputs
        feats = self.convs(x)

        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))

        feats = self.dropout(feats)
        return self.cls_seg(feats)
