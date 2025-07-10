"""Multi-level neck for semantic segmentation."""

import torch
from torch import nn

from geo_deep_learning.models.utils import resize


def xavier_init(
    module: nn.Module,
    gain: float = 1,
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    """Xavier initialization."""
    if distribution not in ["uniform", "normal"]:
        msg = f"Invalid distribution: {distribution}"
        raise ValueError(msg)
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    """ConvModule."""

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        """Initialize ConvModule."""
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm = None
        if norm_cfg is not None and norm_cfg.get("type") == "BN":
            self.norm = nn.BatchNorm2d(out_channels)

        self.act = None
        if act_cfg is not None and act_cfg.get("type") == "ReLU":
            self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MultiLevelNeck(nn.Module):
    """
    MultiLevelNeck.

    A neck structure connect vit backbone and decoder_heads.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.

    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        scales: list[float] | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        """Initialize MultiLevelNeck."""
        super().__init__()
        if not isinstance(in_channels, list):
            msg = f"in_channels must be a list, but got {type(in_channels)}"
            raise TypeError(msg)
        if not isinstance(out_channels, list):
            msg = f"out_channels must be a list, but got {type(out_channels)}"
            raise TypeError(msg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales or [0.5, 1, 2, 4]
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels, out_channels, strict=False):
            self.lateral_convs.append(
                ConvModule(
                    in_channel,
                    out_channel,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )
        for out_channel in out_channels:
            self.convs.append(
                ConvModule(
                    out_channel,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Forward pass."""
        if len(inputs) != len(self.in_channels):
            msg = (
                f"len(inputs) must be equal to len(in_channels), "
                f"but got {len(inputs)} and {len(self.in_channels)}"
            )
            raise ValueError(msg)
        inputs = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # for len(inputs) not equal to self.num_outs
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(self.num_outs)]

        outs = []

        for i in range(self.num_outs):
            x_resize = resize(inputs[i], scale_factor=self.scales[i], mode="bilinear")
            outs.append(self.convs[i](x_resize))

        return tuple(outs)
