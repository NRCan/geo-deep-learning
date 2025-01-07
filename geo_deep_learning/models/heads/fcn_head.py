import torch
from torch import nn
from typing import Union, List
from models.utils import ConvModule

class FCNHead(nn.Module):
    """
    A simplified FCN (Fully Convolutional Network) head for semantic segmentation.
    Adapted from https://github.com/open-mmlab/mmsegmentation
    """
    
    def __init__(
        self,
        in_channels: int,
        channels: int = 256,
        in_index: int = -1,
        num_convs: int = 2,
        num_classes: int = 19,
        concat_input: bool = False,
        dropout_ratio: float = 0.1,
    ) -> None:
        """
        Args:
            in_channels: Number of input channels from the encoder
            channels: Number of intermediate channels in the head
            num_classes: Number of output classes for segmentation
            num_convs: Number of intermediate convolution layers
            concat_input: Whether to concatenate input features with processed features
            dropout_ratio: Dropout ratio for regularization
        """
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
                inplace=True
            )
        )
        
        for _ in range(num_convs - 1):
            convs.append(
                ConvModule(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    inplace=True
                )
            )
            
        self.convs = nn.Identity() if num_convs == 0 else nn.Sequential(*convs)
        
        if self.concat_input:
            self.conv_cat = ConvModule(
                in_channels + channels,
                channels,
                kernel_size=3,
                padding=1,
                inplace=True
            )
            
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Forward pass of the FCN head.
        
        Args:
            inputs: Either a single tensor or a list of tensors. If a list is provided,
                    the tensor at the index specified by `in_index` will be used as input.
                   
        Returns:
            Tensor of shape (batch_size, num_classes, H, W) containing class logits
        """
        x = inputs[self.in_index] if isinstance(inputs, (list, tuple)) else inputs
        feats = self.convs(x)
        
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        
        feats = self.dropout(feats)
        output = self.cls_seg(feats)
        
        return output