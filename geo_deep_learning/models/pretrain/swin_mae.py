from functools import partial
import torch
import torch.nn as nn
from geo_deep_learning.models.pos_embed import get_2d_sincos_pos_embed
from geo_deep_learning.models.decoders.swin_mae_decoder import SwinMAEDecoder
from geo_deep_learning.models.encoders.swin_mae_encoder import SwinMAEEncoder
from geo_deep_learning.models.necks.swin_mae_neck import SwinMAENeck


class SwinMAEPretrainModel(nn.Module):
    """Masked Autoencoder with Swin Transformer backbone."""

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 4,
        mask_ratio: float = 0.75,
        in_chans: int = 3,
        decoder_embed_dim: int = 512,
        norm_pix_loss: bool = False,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        embed_dim: int = 96,
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: type[nn.Module] | None = None,
        patch_norm: bool = True,
    ) -> None:
        super().__init__()

        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss

        effective_norm = norm_layer or nn.LayerNorm

        self.encoder = SwinMAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=effective_norm,
            patch_norm=patch_norm,
        )
        self.neck = SwinMAENeck(decoder_embed_dim=decoder_embed_dim, norm_layer=effective_norm)
        self.decoder = SwinMAEDecoder(
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
            patch_size=patch_size,
            in_chans=in_chans,
            decoder_embed_dim=decoder_embed_dim,
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        pos_embed = get_2d_sincos_pos_embed(
            self.encoder.pos_embed.shape[-1],
            int(self.encoder.num_patches**0.5),
            cls_token=False,
        )
        self.encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.encoder.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
           
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W) where C is the number of channels
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * self.in_chans)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C) where C is the number of channels
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], self.in_chans, h * p, h * p)
        return imgs

    def forward(self, x: torch.Tensor):
        latent, mask = self.encoder(x)
        latent = self.neck(latent)
        pred = self.decoder(latent)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 

        loss = (loss * mask).sum() / mask.sum() 
        return loss

