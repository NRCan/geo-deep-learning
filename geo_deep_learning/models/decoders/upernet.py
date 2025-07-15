"""UperNet decoder."""

import torch
from torch import nn

from geo_deep_learning.models.utils import PPM, ConvModule


class UperNetDecoder(nn.Module):
    """UperNetDecoder. Adapted from MMSegmentation."""

    def __init__(
        self,
        embed_dim: list[int],
        pool_scales: tuple[int] = (1, 2, 3, 6),
        channels: int = 256,
        *,
        align_corners: bool = True,
        scale_modules: bool = False,
    ) -> None:
        """
        Initialize the UPerNet decoder.

        Args:
            embed_dim: Input embedding dimension for each input feature level.
            pool_scales: Pooling scales used in Pyramid Pooling Module applied
                on the last feature. Defaults to (1, 2, 3, 6).
            channels: Number of channels used in the decoder. Defaults to 256.
            align_corners: Whether to align corners in bilinear interpolation
                rescaling. Defaults to True.
            scale_modules: Whether to apply scale modules to the inputs.
                Required for plain ViT encoders. Defaults to False.

        """
        super().__init__()
        self.scale_modules = scale_modules
        if scale_modules:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim[0], embed_dim[0] // 2, 2, 2),
                nn.BatchNorm2d(embed_dim[0] // 2),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim[0] // 2, embed_dim[0] // 4, 2, 2),
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim[1], embed_dim[1] // 2, 2, 2),
            )
            self.fpn3 = nn.Sequential(nn.Identity())
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            self.embed_dim = [
                embed_dim[0] // 4,
                embed_dim[1] // 2,
                embed_dim[2],
                embed_dim[3],
            ]
        else:
            self.embed_dim = embed_dim

        self.out_channels = channels
        self.channels = channels
        self.align_corners = align_corners
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.embed_dim[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = ConvModule(
            self.embed_dim[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            inplace=True,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for embed_dim_ in self.embed_dim[:-1]:  # skip the top layer
            l_conv = ConvModule(
                embed_dim_,
                self.channels,
                1,
                inplace=False,
            )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.embed_dim) * self.channels,
            self.channels,
            3,
            padding=1,
            inplace=True,
        )

    def psp_forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        if self.scale_modules:
            scaled_inputs = []
            scaled_inputs.append(self.fpn1(inputs[0]))
            scaled_inputs.append(self.fpn2(inputs[1]))
            scaled_inputs.append(self.fpn3(inputs[2]))
            scaled_inputs.append(self.fpn4(inputs[3]))
            inputs = scaled_inputs
        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + torch.nn.functional.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = torch.nn.functional.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        return self.fpn_bottleneck(fpn_outs)


if __name__ == "__main__":
    upernet = UperNetDecoder(
        embed_dim=[256, 512, 1024, 2048],
        pool_scales=(1, 2, 3, 6),
        channels=256,
        align_corners=False,
        scale_modules=False,
    )

    x = [
        torch.randn(5, 256, 16, 16),
        torch.randn(5, 512, 8, 8),
        torch.randn(5, 1024, 4, 4),
        torch.randn(5, 2048, 2, 2),
    ]
    out = upernet(x)
    # print(out.shape)
