from __future__ import annotations

from functools import partial

import torch
from torch import nn
from omegaconf import DictConfig

from geo_deep_learning.models.encoders.swin_unet_encoder import SwinEncoder
from geo_deep_learning.models.necks.swin_unet_neck import IdentityNeck
from geo_deep_learning.models.decoders.swin_unet_decoder import SwinDecoder


class SwinUnetSegmentationModel(nn.Module):
    def __init__(self, 
        *,
        image_size: tuple[int, int],
        in_channels: int,
        patch_size: int,
        window_size: int,
        mlp_ratio: float,
        depths: list[int],
        embed_dim: int,
        num_heads: list[int],
        qkv_bias: bool,
        qk_scale: float | None,
        drop_rate: float,
        drop_path_rate: float,
        attn_drop_rate: float,
        ape: bool,
        patch_norm: bool,
        use_checkpoint: bool,
        final_upsample: str,
        num_classes: int) -> None:
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.encoder = SwinEncoder(
            input_size=image_size[0],
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
        )

        self.neck = IdentityNeck(norm_layer(self.encoder.output_dim))

        self.decoder = SwinDecoder(
            patches_resolution=self.encoder.patches_resolution,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            final_upsample=final_upsample,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, self.in_chans, 1, 1)

        bottleneck, skips = self.encoder(x)
        bottleneck, skips = self.neck(bottleneck, skips)
        return self.decoder(bottleneck, skips)