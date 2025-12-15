from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from geo_deep_learning.models.swin_transformer_unet_skip_expand_decoder_sys import (
    BasicLayer,
    PatchEmbed,
    PatchMerging,
)
from timm.models.layers import trunc_normal_


class SwinEncoder(nn.Module):
    """Encoder portion of Swin UNet."""

    def __init__(
        self,
        *,
        input_size: int | tuple[int, int],
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        depths: Sequence[int],
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: float | None,
        drop_rate: float,
        drop_path_rate: float,
        attn_drop_rate: float,
        norm_layer: type[nn.Module],
        ape: bool,
        patch_norm: bool,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth schedule
        dpr = torch.linspace(0, drop_path_rate, steps=sum(depths)).tolist()

        layers: list[BasicLayer] = []
        cursor = 0
        for stage_idx, stage_depth in enumerate(depths):
            layer = BasicLayer(
                dim=int(embed_dim * 2**stage_idx),
                input_resolution=(
                    self.patches_resolution[0] // 2**stage_idx,
                    self.patches_resolution[1] // 2**stage_idx,
                ),
                depth=stage_depth,
                num_heads=num_heads[stage_idx],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cursor : cursor + stage_depth],
                norm_layer=norm_layer,
                downsample=PatchMerging if stage_idx < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            layers.append(layer)
            cursor += stage_depth

        self.layers = nn.ModuleList(layers)
        self.output_dim = int(embed_dim * 2 ** (self.num_layers - 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return bottleneck tokens and a list of skip tensors for decoder."""
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        skips: list[torch.Tensor] = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)

        return x, skips