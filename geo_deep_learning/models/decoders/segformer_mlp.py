"""SegFormer MLP decoder."""

import torch
import torch.nn.functional as fn
from torch import nn


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768) -> None:
        """Initialize the MLP."""
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.flatten(2).transpose(1, 2).contiguous()
        return self.proj(x)


class Decoder(nn.Module):
    """Decoder for SegFormer."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str = "mit_b2",
        in_channels: list[int] | None = None,
        feature_strides: list[int] | None = None,
        embedding_dim: int = 768,
        num_classes: int = 1,
        dropout_ratio: float = 0.1,
    ) -> None:
        """Initialize the decoder."""
        super().__init__()
        if feature_strides is None:
            feature_strides = [4, 8, 16, 32]
        if in_channels is None:
            in_channels = [64, 128, 320, 512]
        if encoder == "mit_b0":
            in_channels = [32, 64, 160, 256]
            embedding_dim = 256
        elif encoder == "mit_b1":
            embedding_dim = 256
        if len(feature_strides) != len(in_channels):
            msg = "feature_strides and in_channels must have the same length"
            raise ValueError(msg)
        if min(feature_strides) != feature_strides[0]:
            msg = "The minimum feature stride must be the first element"
            raise ValueError(msg)

        self.num_classes = num_classes
        self.in_channels = in_channels
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = (
            self.in_channels
        )

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dim * 4,
                out_channels=embedding_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape

        _c4 = (
            self.linear_c4(c4)
            .permute(0, 2, 1)
            .reshape(n, -1, c4.shape[2], c4.shape[3])
            .contiguous()
        )
        _c4 = fn.interpolate(
            input=_c4,
            size=c1.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        _c3 = (
            self.linear_c3(c3)
            .permute(0, 2, 1)
            .reshape(n, -1, c3.shape[2], c3.shape[3])
            .contiguous()
        )
        _c3 = fn.interpolate(
            input=_c3,
            size=c1.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        _c2 = (
            self.linear_c2(c2)
            .permute(0, 2, 1)
            .reshape(n, -1, c2.shape[2], c2.shape[3])
            .contiguous()
        )
        _c2 = fn.interpolate(
            input=_c2,
            size=c1.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        _c1 = (
            self.linear_c1(c1)
            .permute(0, 2, 1)
            .reshape(n, -1, c1.shape[2], c1.shape[3])
            .contiguous()
        )
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        return self.linear_pred(x)
