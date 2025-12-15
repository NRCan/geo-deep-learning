"""
Masked Autoencoder (MAE) Encoder
Vision Transformer encoder with random masking capability
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block



class MAEEncoder(nn.Module):
    """
    MAE Encoder with Vision Transformer backbone
    Handles patch embedding, positional encoding, and random masking
    """
    
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        mask_ratio=0.75
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), 
            requires_grad=False
        )  # fixed sin-cos embedding
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        
    
    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        
        Args:
            x: [N, L, D], sequence
            
        Returns:
            x_masked: [N, L_keep, D], masked sequence
            mask: [N, L], binary mask (0 is keep, 1 is remove)
            ids_restore: [N, L], indices to restore original order
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x):
        """
        Forward pass of MAE encoder
        
        Args:
            x: [N, C, H, W], input images
            
        Returns:
            x: [N, L_keep + 1, D], encoded tokens (with cls token)
            mask: [N, L], binary mask
            ids_restore: [N, L], indices to restore original order
        """
        # Embed patches
        x = self.patch_embed(x)
        
        # Add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking: length -> length * (1 - mask_ratio)
        x, mask, ids_restore = self.random_masking(x)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore

