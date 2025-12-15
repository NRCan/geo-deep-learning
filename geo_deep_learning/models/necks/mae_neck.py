"""
Masked Autoencoder (MAE) Neck
Connects encoder and decoder by projecting encoder features to decoder dimension
"""

import torch
import torch.nn as nn


class MAENeck(nn.Module):
    """
    MAE Neck - Projects encoder output to decoder dimension
    This is a simple linear projection layer that connects the encoder and decoder
    """
    
    def __init__(self, embed_dim=1024, decoder_embed_dim=512):
        """
        Args:
            embed_dim: Encoder embedding dimension
            decoder_embed_dim: Decoder embedding dimension
        """
        super().__init__()
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

    
    def forward(self, x):
        """
        Forward pass of MAE neck
        
        Args:
            x: [N, L, embed_dim], encoder output
            
        Returns:
            x: [N, L, decoder_embed_dim], projected features
        """
        x = self.decoder_embed(x)
        return x

