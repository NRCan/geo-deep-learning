"""
Swin Transformer Encoder
Hierarchical vision transformer encoder using shifted windows
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from geo_deep_learning.models.swin_transformer_unet_skip_expand_decoder_sys import (
    PatchEmbed,
    BasicLayer,
    PatchMerging
)


class SwinEncoder(nn.Module):
    """
    Swin Transformer Encoder
    
    A hierarchical vision transformer that builds representations through 
    shifted window based self-attention at multiple scales.
    """

    def __init__(
        self,
        input_size=512,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs
    ):
        """
        Args:
            input_size: Input image size
            patch_size: Patch size for patch embedding
            in_chans: Number of input channels
            embed_dim: Patch embedding dimension
            depths: Depths of each Swin Transformer stage
            num_heads: Number of attention heads in different layers
            window_size: Window size for shifted window attention
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: If True, add a learnable bias to query, key, value
            qk_scale: Override default qk scale of head_dim ** -0.5 if set
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            norm_layer: Normalization layer
            ape: If True, add absolute position embedding to the patch embedding
            patch_norm: If True, add normalization after patch embedding
            use_checkpoint: Whether to use checkpointing to save memory
        """
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights for Linear and LayerNorm layers"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        """
        Forward pass extracting hierarchical features
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            x: Final encoded features (B, L, C)
            x_downsample: List of features from each stage
        """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        
        x = self.norm(x)  # B L C
        return x, x_downsample
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (final_features, intermediate_features)
        """
        return self.forward_features(x)

