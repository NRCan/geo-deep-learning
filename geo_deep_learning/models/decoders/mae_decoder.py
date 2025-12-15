"""
Masked Autoencoder (MAE) Decoder
Vision Transformer decoder for reconstructing masked patches
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block



class MAEDecoder(nn.Module):
    """
    MAE Decoder with Vision Transformer backbone
    Reconstructs masked patches from encoder latent representation
    """
    
    def __init__(
        self,
        num_patches,
        patch_size=16,
        in_chans=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )  # fixed sin-cos embedding
        
        # Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Prediction head - projects back to pixel space
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size**2 * in_chans,
            bias=True
        )
     

    def forward(self, x, ids_restore):
        """
        Forward pass of MAE decoder
        
        Args:
            x: [N, L_keep + 1, decoder_embed_dim], encoder output (projected through neck)
            ids_restore: [N, L], indices to restore original patch order
            
        Returns:
            x: [N, L, patch_size**2 * in_chans], reconstructed patches
        """
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], 
            ids_restore.shape[1] + 1 - x.shape[1], 
            1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor projection
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x

