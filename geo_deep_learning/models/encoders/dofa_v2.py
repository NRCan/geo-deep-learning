"""Dynamic One-For-All (DOFA) v2 encoder."""

import torch
import torch.nn.functional as fn
from timm.models.vision_transformer import Block
from torch import Tensor, nn


def position_embedding(embed_dim: int, pos: Tensor) -> Tensor:
    """
    Compute 1D sine/cosine position embedding.

    Args:
        embed_dim: Output dimension D for each position. Must be even.
        pos: Positions to be encoded, shape (M,).

    Returns:
        Position embeddings of shape (M, D).

    """
    if embed_dim % 2 != 0:
        msg = "embed_dim must be even"
        raise ValueError(msg)

    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    return torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)


class FCResLayer(nn.Module):
    """Fully-connected residual layer."""

    def __init__(self, linear_size: int = 128) -> None:
        """Initialize FCResLayer."""
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        return x + y


class TransformerWeightGenerator(nn.Module):
    """Simplified dynamic weight generator for DOFA v2."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        """Initialize TransformerWeightGenerator."""
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Linear layers for weight and bias generation
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        # Learnable tokens
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))
        # Initialize
        nn.init.normal_(self.weight_tokens, std=0.02)
        nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(transformer_output[-1])
        return weights, bias


class DOFAv2Embedding(nn.Module):
    """Dynamic One-For-All v2 embedding layer."""

    def __init__(
        self,
        dynamic_embed_dim: int = 128,
        kernel_size: int = 14,
        embed_dim: int = 768,
        *,
        convert_to_16: bool = False,
    ) -> None:
        """Initialize DOFAv2Embedding."""
        super().__init__()
        self.dynamic_embed_dim = dynamic_embed_dim
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.convert_to_16 = convert_to_16

        self._num_kernel = kernel_size * kernel_size * embed_dim

        self.weight_generator = TransformerWeightGenerator(
            input_dim=dynamic_embed_dim,
            output_dim=self._num_kernel,
            embed_dim=embed_dim,
        )

        self.fclayer = FCResLayer(dynamic_embed_dim)
        self.scaler = 0.01

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def forward(self, x: Tensor, wavelengths: Tensor) -> Tensor:
        """Forward pass."""
        _, in_channels, _, _ = x.shape
        # Generate position embeddings from wavelengths
        waves = position_embedding(self.dynamic_embed_dim, wavelengths * 1000)
        waves = self.fclayer(waves)  # Shape: [C, dynamic_embed_dim]
        # Generate dynamic weights and bias
        weight, bias = self.weight_generator(waves)
        # Reshape weights for convolution
        dynamic_weight = weight.view(
            in_channels,
            self.kernel_size,
            self.kernel_size,
            self.embed_dim,
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])
        weights = dynamic_weight * self.scaler
        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler
        # Handle patch size conversion if needed
        if self.convert_to_16:
            weights = fn.interpolate(
                weights,
                size=(16, 16),
                mode="bicubic",
                align_corners=False,
            )
            stride = 16
        else:
            stride = self.kernel_size
        # Apply dynamic convolution
        x = fn.conv2d(x, weights, bias=bias, stride=stride, padding=1, dilation=1)
        # Flatten spatial dimensions
        return x.flatten(2).transpose(1, 2)  # [B, L, D]


class DOFAv2(nn.Module):
    """Dynamic One-For-All v2 encoder."""

    def __init__(  # noqa: PLR0913
        self,
        encoder_name: str = "dofa_base",
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 14,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        out_indices: list[int] | None = None,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: float = 1e-5,
        *,
        convert_patch_to_16: bool = False,
        pretrained: bool = True,
    ) -> None:
        """Initialize DOFAv2."""
        super().__init__()

        self.encoder_name = encoder_name
        self.pretrained = pretrained
        # Handle image size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        effective_patch_size = 16 if convert_patch_to_16 else patch_size
        self.num_patches = (img_size[0] // effective_patch_size) * (
            img_size[1] // effective_patch_size
        )
        if out_indices is None:
            out_indices = [depth - 1]
        self.out_indices = out_indices

        self.patch_embed = DOFAv2Embedding(
            dynamic_embed_dim=128,
            kernel_size=patch_size,
            embed_dim=embed_dim,
            convert_to_16=convert_patch_to_16,
        )
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False,
        )

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ],
        )

        # Final norm
        self.norm = norm_layer(embed_dim)

        self.init_weights()
        if self.pretrained:
            self.load_pretrained_weights()

    def init_weights(self) -> None:
        """Initialize weights."""
        # Initialize position embedding with sin-cos
        pos_embed = self.get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        # Initialize class token
        nn.init.normal_(self.cls_token, std=0.02)

    def load_pretrained_weights(self) -> tuple[list[str], list[str]]:  # noqa: C901
        """Load pretrained weights."""
        # Download from HuggingFace
        if self.encoder_name == "dofa_base":
            url = "https://hf.co/earthflow/DOFA/resolve/main/dofav2_vit_base_e150.pth"
        elif self.encoder_name == "dofa_large":
            url = "https://hf.co/earthflow/DOFA/resolve/main/dofav2_vit_large_e150.pth"
        else:
            msg = f"Unknown model name: {self.encoder_name}"
            raise ValueError(msg)
        state_dict = torch.hub.load_state_dict_from_url(
            url,
            progress=True,
            map_location="cpu",
            weights_only=True,
        )
        # Handle state dict format differences
        if "model" in state_dict:
            state_dict = state_dict["model"]

        # Create mapping from old keys to new keys
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key
            if key.startswith("model."):
                new_key = key[6:]
                if new_key.startswith(("blocks.", "norm.")) or new_key in {
                    "cls_token",
                    "pos_embed",
                }:
                    pass
                else:
                    continue
            elif key.startswith("patch_embed."):
                new_key = key
            new_state_dict[new_key] = value
        # Handle position embedding size mismatch
        if (
            "pos_embed" in new_state_dict
            and self.pos_embed.shape != new_state_dict["pos_embed"].shape
        ):
            new_state_dict["pos_embed"] = self._resize_pos_embed(
                new_state_dict["pos_embed"],
                self.num_patches,
                self.num_patches + 1,
            )

        # Load state dict
        missing_keys, unexpected_keys = self.load_state_dict(
            new_state_dict,
            strict=False,
        )
        expected_missing = {"head.weight", "head.bias"}
        actual_missing = set(missing_keys) - expected_missing
        if actual_missing:
            msg = f"Missing required keys in state dict: {actual_missing}"
            raise RuntimeError(msg)
        if unexpected_keys:
            msg = f"Unexpected keys in state dict: {unexpected_keys}"
            raise RuntimeError(msg)
        return missing_keys, unexpected_keys

    def _resize_pos_embed(
        self,
        pos_embed: Tensor,
        num_patches: int,
        num_tokens: int,
    ) -> Tensor:
        """Resize position embeddings to match model dimensions."""
        # pos_embed shape: [1, num_tokens_old, embed_dim]
        if pos_embed.shape[1] == num_tokens:
            return pos_embed

        # Separate class token and position embeddings
        cls_token = pos_embed[:, :1, :]
        pos_tokens = pos_embed[:, 1:, :]

        # Calculate grid sizes
        old_size = int(pos_tokens.shape[1] ** 0.5)
        new_size = int(num_patches**0.5)

        if old_size != new_size:
            # Reshape to 2D grid
            pos_tokens = pos_tokens.reshape(1, old_size, old_size, -1).permute(
                0,
                3,
                1,
                2,
            )
            # Interpolate
            pos_tokens = fn.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )

            # Reshape back
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                1,
                -1,
                pos_embed.shape[-1],
            )

        # Concatenate class token and resized position embeddings
        return torch.cat([cls_token, pos_tokens], dim=1)

    @staticmethod
    def get_2d_sincos_pos_embed(
        embed_dim: int,
        grid_size: int,
        *,
        cls_token: bool = False,
    ) -> Tensor:
        """Generate 2D sin-cos position embedding."""
        grid_h = grid_w = grid_size
        grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
        grid = torch.stack(grid, dim=0).reshape([2, 1, grid_h, grid_w])
        pos_embed = DOFAv2.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
        return pos_embed

    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: Tensor) -> Tensor:
        """Generate position embedding from grid."""
        if embed_dim % 2 != 0:
            msg = "embed_dim must be even"
            raise ValueError(msg)
        emb_h = DOFAv2.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = DOFAv2.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        return torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)

    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: Tensor) -> Tensor:
        """Generate 1D sin-cos embedding from positions."""
        if embed_dim % 2 != 0:
            msg = "embed_dim must be even"
            raise ValueError(msg)
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)
        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum("m,d->md", pos, omega)  # (M, D/2)
        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)
        return torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

    def forward_features(self, x: Tensor, wavelengths: Tensor) -> list[Tensor]:
        """Forward pass extracting features at specified layers."""
        expected_ndim = 2
        if wavelengths.dim() == expected_ndim:
            if not torch.allclose(wavelengths, wavelengths[0:1].expand_as(wavelengths)):
                msg = "DOFA cannot handle different wavelengths within a batch"
                raise ValueError(msg)
            wavelengths = wavelengths[0]

        # Patch embedding
        x = self.patch_embed(x, wavelengths)  # [B, L, D]

        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]

        # Prepend class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply position dropout
        x = self.pos_drop(x)

        # Collect features
        features = []

        # Apply transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                # Extract spatial features (remove cls token)
                feat = x[:, 1:, :]

                # Reshape to spatial format
                batch_size, length, channels = feat.shape
                height = width = int(length**0.5)
                feat = feat.reshape(batch_size, height, width, channels).permute(
                    0,
                    3,
                    1,
                    2,
                )
                features.append(feat)
        # Apply final norm if last layer is requested
        if (self.depth - 1) in self.out_indices and len(features) < len(
            self.out_indices,
        ):
            x = self.norm(x)
            feat = x[:, 1:, :]
            batch_size, length, channels = feat.shape
            height = width = int(length**0.5)
            feat = feat.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
            features.append(feat)
        return features

    def forward(self, x: Tensor, wavelengths: Tensor) -> list[Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input image tensor of shape [B, C, H, W].
            wavelengths: Wavelengths of each spectral band (μm).

        Returns:
            List of feature maps from specified layers.

        """
        return self.forward_features(x, wavelengths)


def create_dofa_base(
    img_size: int | tuple[int, int] = 224,
    out_indices: int | list[int] | None = None,
    *,
    pretrained: bool = True,
    **kwargs: object,
) -> DOFAv2:
    """
    Create DOFA base model.

    Args:
        img_size: Input image size.
        out_indices: Layers to extract features from.
        pretrained: Whether to load pretrained weights.
        **kwargs: Additional arguments for DOFA.

    Returns:
        DOFA base model instance.

    """
    return DOFAv2(
        encoder_name="dofa_base",
        img_size=img_size,
        patch_size=14,
        embed_dim=768,
        num_heads=12,
        depth=12,
        out_indices=out_indices or [4, 6, 10, 11],
        pretrained=pretrained,
        **kwargs,
    )


def create_dofa_large(
    img_size: int | tuple[int, int] = 224,
    out_indices: int | list[int] | None = None,
    *,
    pretrained: bool = True,
    **kwargs: object,
) -> DOFAv2:
    """
    Create DOFA large model.

    Args:
        img_size: Input image size.
        out_indices: Layers to extract features from.
        pretrained: Whether to load pretrained weights.
        **kwargs: Additional arguments for DOFA.

    Returns:
        DOFA large model instance.

    """
    return DOFAv2(
        encoder_name="dofa_large",
        img_size=img_size,
        patch_size=14,
        embed_dim=1024,
        num_heads=16,
        depth=24,
        out_indices=out_indices or [5, 9, 15, 21],
        pretrained=pretrained,
        **kwargs,
    )
