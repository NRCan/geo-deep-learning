from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from einops import rearrange
from torch import nn

from geo_deep_learning.models.swin_transform import PatchEmbedding, BasicBlock


class SwinMAEEncoder(nn.Module):
    """Encoder for Swin Masked Autoencoder.

    Applies patch embedding, window masking, and stacked Swin transformer blocks.
    """

    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        mask_ratio: float,
        in_chans: int,
        embed_dim: int,
        depths: Sequence[int],
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        norm_layer: type[nn.Module] | None,
        patch_norm: bool,
    ) -> None:
        super().__init__()

        self.mask_ratio = mask_ratio
        self.window_size = window_size
        self.num_layers = len(depths)
        self.num_patches = (img_size // patch_size) ** 2

        effective_norm_layer = norm_layer or nn.LayerNorm

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
            norm_layer=effective_norm_layer if patch_norm else None,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
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
            norm_layer=effective_norm_layer,
        )

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
        for stage_idx, _ in enumerate(depths):
            layer = BasicBlock(
                index=stage_idx,
                depths=tuple(depths),
                embed_dim=embed_dim,
                num_heads=tuple(num_heads),
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                patch_merging=False if stage_idx == len(depths) - 1 else True
            )
            layers.append(layer)
        return layers

    def window_masking(
        self,
        x: torch.Tensor,
        *,
        r: int = 4,
        remove: bool = False,
        mask_len_sparse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply window-based masking strategy.

        Args:
            x: Tensor with shape [B, H, W, C].
            r: Window size for grouping patches.
            remove: Whether to remove the masked patches.
            mask_len_sparse: Whether to return sparse mask length.

        Returns:
            Masked tensor and associated mask.
        """
        x = rearrange(x, "B H W C -> B (H W) C")
        batch_size, num_tokens, channels = x.shape
        assert int(num_tokens**0.5 / r) == num_tokens**0.5 / r
        d = int(num_tokens**0.5 // r)

        noise = torch.rand(batch_size, d**2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, : int(d**2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode="floor") * d * r**2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(num_tokens**0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(num_tokens), axis=0).repeat(batch_size, axis=0)
        index_mask = np.zeros([batch_size, int(num_tokens - index_keep.shape[-1])], dtype=np.int64)
        for i in range(batch_size):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.detach().cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([batch_size, d**2], device=x.device)
            mask[:, : sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([batch_size, num_tokens], device=x.device)
            mask[:, : index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, channels))
            x_masked = rearrange(x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)
            for i in range(batch_size):
                x_masked[i, index_mask.detach().cpu().numpy()[i, :], :] = self.mask_token
            x_masked = rearrange(x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        x, mask = self.window_masking(x, remove=False, mask_len_sparse=False)
        for layer in self.layers:
            x = layer(x)
        return x, mask

