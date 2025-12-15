from __future__ import annotations

from torch import nn

from geo_deep_learning.models.swin_transform import PatchExpanding


class SwinMAENeck(nn.Module):
    """Neck component for Swin MAE bridging encoder and decoder stages."""

    def __init__(self, *, decoder_embed_dim: int, norm_layer: type[nn.Module] | None) -> None:
        super().__init__()
        effective_norm = norm_layer or nn.LayerNorm
        self.patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=effective_norm)

    def forward(self, x):
        return self.patch_expanding(x)

