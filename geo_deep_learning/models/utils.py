"""Utility functions for models."""

import warnings

import torch
import torch.nn.functional as f
from torch import nn


class ConvModule(nn.Module):
    """Convolution module."""

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        dilation: int = 1,
        stride: int = 1,
        *,
        inplace: bool = False,
        transpose: bool = False,
        scale_factor: int | None = None,
    ) -> None:
        """Initialize the convolution module."""
        super().__init__()

        kind = "Transpose" if transpose else ""

        conv_name = f"Conv{kind}2d"

        if transpose:
            stride = scale_factor
            padding = (kernel_size - scale_factor) // 2

        conv_template = getattr(nn, conv_name)
        self.conv = conv_template(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return self.act(self.norm(self.conv(x)))


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(
        self,
        pool_scales: tuple[int, ...],
        in_channels: int,
        channels: int,
        *,
        align_corners: bool,
    ) -> None:
        """Initialize the Pooling Pyramid Module."""
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels

        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(self.in_channels, self.channels, 1, inplace=True),
                ),
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = torch.nn.functional.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


def resize(  # noqa: PLR0913
    input_: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "nearest",
    *,
    align_corners: bool | None = None,
    warning: bool = True,
) -> torch.Tensor:
    """Resize a tensor."""
    if scale_factor is not None:
        h, w = input_.shape[2:]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        size = (new_h, new_w)
        scale_factor = None
    if warning and size is not None and align_corners:
        input_h, input_w = tuple(int(x) for x in input_.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if output_h > input_h or (
            output_w > input_w
            and output_h > 1
            and output_w > 1
            and input_h > 1
            and input_w > 1
            and (output_h - 1) % (input_h - 1)
            and (output_w - 1) % (input_w - 1)
        ):
            warnings.warn(
                f"When align_corners={align_corners}, "
                "the output would more aligned if "
                f"input size {(input_h, input_w)} is `x+1` and "
                f"out size {(output_h, output_w)} is `nx+1`",
                stacklevel=2,
            )
    return f.interpolate(
        input_,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


def patch_first_conv(
    model: nn.Module,
    new_in_channels: int,
    default_in_channels: int = 3,
    *,
    pretrained: bool = True,
) -> None:
    """Change first convolution layer input channels."""
    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(
                module.out_channels,
                new_in_channels // module.groups,
                *module.kernel_size,
            ),
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(
            module.out_channels,
            new_in_channels // module.groups,
            *module.kernel_size,
        )

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)
