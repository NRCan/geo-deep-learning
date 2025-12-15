from __future__ import annotations

from typing import Sequence

from einops import rearrange
from torch import nn

from geo_deep_learning.models.swin_transform import BasicBlockUp


class SwinMAEDecoder(nn.Module):
    """Decoder for Swin Masked Autoencoder."""

    def __init__(
        self,
        *,
        depths: Sequence[int],
        embed_dim: int,
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        norm_layer: type[nn.Module] | None,
        patch_size: int,
        in_chans: int,
        decoder_embed_dim: int,
    ) -> None:
        super().__init__()

        effective_norm = norm_layer or nn.LayerNorm
        self.layers = self._build_layers(
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=effective_norm,
        )
        self.norm = effective_norm(embed_dim)
        self.projection = nn.Linear(decoder_embed_dim // 8, patch_size**2 * in_chans, bias=True)

    @staticmethod
    def _build_layers(
        *,
        depths: Sequence[int],
        embed_dim: int,
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        norm_layer: type[nn.Module],
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        num_layers = len(depths)
        for index in range(num_layers - 1):
            layer = BasicBlockUp(
                index=index,
                depths=tuple(depths),
                embed_dim=embed_dim,
                num_heads=tuple(num_heads),
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path=drop_path_rate,
                patch_expanding=index < num_layers - 2,
                norm_layer=norm_layer,
            )
            layers.append(layer)
        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = rearrange(x, "B H W C -> B (H W) C")
        return self.projection(x)

