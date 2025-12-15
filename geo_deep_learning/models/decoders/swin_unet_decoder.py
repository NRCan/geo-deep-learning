from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from geo_deep_learning.models.swin_transformer_unet_skip_expand_decoder_sys import (
    BasicLayer_up,
    FinalPatchExpand_X4,
    PatchExpand,
)
from timm.models.layers import trunc_normal_


class SwinDecoder(nn.Module):
    """Decoder/up-sampling path for Swin UNet."""

    def __init__(
        self,
        *,
        patches_resolution: tuple[int, int],
        embed_dim: int,
        depths: Sequence[int],
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: float | None,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        norm_layer: type[nn.Module],
        use_checkpoint: bool,
        final_upsample: str,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.patches_resolution = patches_resolution
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.final_upsample = final_upsample
        self.num_classes = num_classes

        dpr = torch.linspace(0, drop_path_rate, steps=sum(depths)).tolist()

        layers: list[nn.Module] = []
        concat_layers: list[nn.Module] = []
        cursor = 0
        for idx in range(self.num_stages):
            stage_dim = int(embed_dim * 2 ** (self.num_stages - 1 - idx))

            concat = (
                nn.Linear(2 * stage_dim, stage_dim, bias=False)
                if idx > 0
                else nn.Identity()
            )
            concat_layers.append(concat)

            if idx == 0:
                layer = PatchExpand(
                    input_resolution=(
                        patches_resolution[0] // 2 ** (self.num_stages - 1 - idx),
                        patches_resolution[1] // 2 ** (self.num_stages - 1 - idx),
                    ),
                    dim=stage_dim,
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
            else:
                layer = BasicLayer_up(
                    dim=stage_dim,
                    input_resolution=(
                        patches_resolution[0] // 2 ** (self.num_stages - 1 - idx),
                        patches_resolution[1] // 2 ** (self.num_stages - 1 - idx),
                    ),
                    depth=depths[self.num_stages - 1 - idx],
                    num_heads=num_heads[self.num_stages - 1 - idx],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cursor : cursor + depths[self.num_stages - 1 - idx]],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if idx < self.num_stages - 1 else None,
                    use_checkpoint=use_checkpoint,
                )
                cursor += depths[self.num_stages - 1 - idx]

            layers.append(layer)

        self.layers_up = nn.ModuleList(layers)
        self.concat_back_dim = nn.ModuleList(concat_layers)
        self.head_norm = norm_layer(embed_dim)

        if final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                dim_scale=4,
                dim=embed_dim,
            )
            self.output = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=num_classes,
                kernel_size=1,
                bias=False,
            )
        else:
            raise ValueError(f"Unsupported final_upsample: {final_upsample}")

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: list[torch.Tensor],
    ) -> torch.Tensor:
        x = bottleneck
        for idx, layer in enumerate(self.layers_up):
            if idx == 0:
                x = layer(x)
            else:
                skip = skips[self.num_stages - 1 - idx]
                x = torch.cat([x, skip], dim=-1)
                x = self.concat_back_dim[idx](x)
                x = layer(x)

        x = self.head_norm(x)

        #up_x4
        h, w = self.patches_resolution  # type: ignore[attr-defined]
        B, L, C = x.shape
    
        if L != h * w:
            raise RuntimeError("Token count mismatch in decoder output.")

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * h, 4 * w, -1).permute(0, 3, 1, 2).contiguous()
            x = self.output(x)

        return x