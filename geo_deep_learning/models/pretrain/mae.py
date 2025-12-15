"""
Masked Autoencoder (MAE) with Vision Transformer backbone
Complete model integrating encoder, neck, and decoder components
"""

from functools import partial
import torch
import torch.nn as nn
from omegaconf import DictConfig

from geo_deep_learning.models.encoders.mae_encoder import MAEEncoder
from geo_deep_learning.models.necks.mae_neck import MAENeck
from geo_deep_learning.models.decoders.mae_decoder import MAEDecoder
from geo_deep_learning.models.pos_embed import get_2d_sincos_pos_embed


class MAEPretrainModel(nn.Module):
    """
    Masked Autoencoder with Vision Transformer backbone
    Integrates encoder, neck, and decoder for self-supervised pretraining
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] | None = None,
        norm_pix_loss: bool = False,
    ) -> None:
        """
        Args:
            img_size: Input image size
            patch_size: Patch size for tokenization
            mask_ratio: Ratio of patches to mask (0-1)
            in_chans: Number of input channels
            embed_dim: Encoder embedding dimension
            depth: Number of encoder transformer blocks
            num_heads: Number of attention heads in encoder
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder transformer blocks
            decoder_num_heads: Number of attention heads in decoder
            mlp_ratio: MLP hidden dimension ratio
            norm_layer: Normalization layer type
            norm_pix_loss: Whether to normalize pixels for loss calculation
        """
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss

        effective_norm = norm_layer or nn.LayerNorm

        # Initialize encoder
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=effective_norm,
            mask_ratio=mask_ratio,
        )

        # Initialize neck (projection from encoder to decoder dimension)
        self.neck = MAENeck(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
        )

        # Initialize decoder
        self.decoder = MAEDecoder(
            num_patches=self.encoder.patch_embed.num_patches,
            patch_size=patch_size,
            in_chans=in_chans,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=effective_norm,
        )

        self.initialize_weights()
    
    def initialize_weights(self):
        """
        Initialize weights for all components (encoder, neck, decoder)
        Note: Individual components already initialize their own weights in their __init__,
        but this method ensures consistent initialization across the full model
        """
        # Initialize encoder positional embeddings with sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.encoder.pos_embed.shape[-1],
            int(self.encoder.patch_embed.num_patches**.5),
            cls_token=True
        )
        self.encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize decoder positional embeddings with sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder.decoder_pos_embed.shape[-1],
            int(self.decoder.num_patches**.5),
            cls_token=True
        )
        self.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize tokens
        torch.nn.init.normal_(self.encoder.cls_token, std=.02)
        torch.nn.init.normal_(self.decoder.mask_token, std=.02)
        
        # Initialize all Linear and LayerNorm layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights for Linear and LayerNorm layers"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Convert images to patches
        
        Args:
            imgs: (N, C, H, W) where C is the number of channels
            
        Returns:
            x: (N, L, patch_size**2 * C) where L is the number of patches
        """
        p = self.encoder.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * self.in_chans)
        return x

    def unpatchify(self, x):
        """
        Convert patches back to images
        
        Args:
            x: (N, L, patch_size**2 * C) where C is the number of channels
            
        Returns:
            imgs: (N, C, H, W)
        """
        p = self.encoder.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], self.in_chans, h * p, h * p)
        return imgs

    def forward(self, imgs: torch.Tensor):
        """
        Forward pass: encode -> project -> decode -> compute loss
        
        Args:
            imgs: [N, C, H, W], input images
            
        Returns:
            loss: scalar, reconstruction loss on masked patches
            pred: [N, L, patch_size**2 * C], predicted patches
            mask: [N, L], binary mask (0 is keep, 1 is remove)
        """
        # Encode with masking
        latent, mask, ids_restore = self.encoder(imgs)
        
        # Project to decoder dimension
        latent = self.neck(latent)
        
        # Decode
        pred = self.decoder(latent, ids_restore)
        
        # Compute loss
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask

    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss on masked patches
        
        Args:
            imgs: [N, C, H, W], original images
            pred: [N, L, patch_size**2 * C], predicted patches
            mask: [N, L], binary mask (0 is keep, 1 is remove)
            
        Returns:
            loss: scalar, mean loss on removed patches
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


def MAE(cfg: DictConfig) -> MAEPretrainModel:
    """
    Factory function to create MAE model from config
    
    Args:
        cfg: Configuration object with model parameters
        
    Returns:
        MAEPretrainModel instance
    """
    model = MAEPretrainModel(
        img_size=cfg.model.get('img_size', 512),
        patch_size=cfg.model.patch_size,
        mask_ratio=cfg.model.get('mask_ratio', 0.75),
        in_chans=cfg.model.get('in_chans', 3),
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        decoder_embed_dim=cfg.model.decoder_embed_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        mlp_ratio=cfg.model.get('mlp_ratio', 4.0),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=cfg.model.get('norm_pix_loss', False),
    )
    return model
