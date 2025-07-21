"""Dynamic One-For-All (DOFA) v2 encoder."""

import math

import torch
import torch.nn.functional as fn
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn
from torch.nn import init


def position_embedding(embed_dim: int, pos: Tensor) -> Tensor:
    """
    Compute the 1D sine/cosine position embedding.

    Args:
        embed_dim: Output dimension D for each position. Must be even.
        pos: A list of positions to be encoded, of size (M,).

    Returns:
        Position embeddings of size (M, D).

    Raises:
        ValueError: If embed_dim is not even.

    """
    if embed_dim % 2 != 0:
        msg = "embed_dim must be even"
        raise ValueError(msg)

    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    return torch.cat([emb_sin, emb_cos], dim=1)


def dofa_encoder_base() -> dict[str, Tensor]:
    """Load DOFA encoder base model."""
    url: str = "https://hf.co/earthflow/DOFA/resolve/main/dofav2_vit_base_e150.pth"
    return torch.hub.load_state_dict_from_url(
        url,
        progress=True,
        map_location="cpu",
        weights_only=True,
    )


def dofa_encoder_large() -> dict[str, Tensor]:
    """Load DOFA encoder large model."""
    url: str = "https://hf.co/earthflow/DOFA/resolve/main/dofav2_vit_large_e150.pth"
    return torch.hub.load_state_dict_from_url(
        url,
        progress=True,
        map_location="cpu",
        weights_only=True,
    )


class TransformerWeightGenerator(nn.Module):
    """Dynamic weight generator for DOFA."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        """
        Initialize a new TransformerWeightGenerator instance.

        Args:
            input_dim: Input dimensions.
            output_dim: Output dimensions.
            embed_dim: Embedding dimensions.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.

        """
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=0.0,  # No dropout for better performance
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

        # Initialize with normal distribution
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input position embeddings of shape (seq_len, batch, input_dim).

        Returns:
            Tuple of (weights, bias).

        """
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)

        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(transformer_output[-1])

        return weights, bias


class FCResLayer(nn.Module):
    """Fully-connected residual layer."""

    def __init__(self, linear_size: int = 128) -> None:
        """
        Initialize a new FCResLayer instance.

        Args:
            linear_size: Size of linear layers.

        """
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with residual connection.

        """
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out: Tensor = x + y
        return out


class DOFAEmbedding(nn.Module):
    """Dynamic One-For-All (DOFA) embedding."""

    def __init__(
        self,
        wv_planes: int,
        dynamic_embed_dim: int = 128,
        kernel_size: int = 14,
        embed_dim: int = 768,
        *,
        convert_patch_14_to_16: bool = False,
    ) -> None:
        """
        Initialize a new DOFAEmbedding instance.

        Args:
            wv_planes: Number of wavelength planes.
            dynamic_embed_dim: Dimensions of dynamic weight generator.
            kernel_size: Kernel size of the convolution (14 for v2).
            embed_dim: Output embedding dimensions.
            convert_patch_14_to_16: If True, convert patch size from 14 to 16.

        """
        super().__init__()
        self.wv_planes = wv_planes
        self.dynamic_embed_dim = dynamic_embed_dim
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.convert_patch_14_to_16 = convert_patch_14_to_16
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.patch_size = (kernel_size, kernel_size)

        self.weight_generator = TransformerWeightGenerator(
            wv_planes,
            self._num_kernel,
            embed_dim,
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _init_weight(self, m: nn.Module) -> None:
        """Initialize weights of a single layer."""
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        """Initialize all weights in the module."""
        self.weight_generator.apply(self._init_weight)
        self.fclayer.apply(self._init_weight)

    def forward(self, x: Tensor, wavelengths: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the embedding layer.

        Args:
            x: Input image tensor of shape [B, C, H, W].
            wavelengths: Wavelengths of each spectral band (μm).

        Returns:
            Tuple of (embedded features [B, L, D], processed wavelengths).

        """
        inplanes = wavelengths.size(0)

        # Generate position embeddings from wavelengths
        waves = position_embedding(self.dynamic_embed_dim, wavelengths * 1000)
        waves = self.fclayer(waves)

        # Generate dynamic weights and bias
        weight, bias = self.weight_generator(waves)

        # Reshape weights for convolution
        dynamic_weight = weight.view(
            inplanes,
            self.kernel_size,
            self.kernel_size,
            self.embed_dim,
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        # Handle patch size conversion if needed
        if self.convert_patch_14_to_16:
            _kernel_size = 14
            if self.kernel_size != _kernel_size:
                msg = f"convert_patch_14_to_16 works with kernel_size={_kernel_size}"
                raise ValueError(msg)

            weights = fn.interpolate(
                weights,
                size=(16, 16),
                mode="bicubic",
                align_corners=False,
            )
            stride = 16
        else:
            stride = self.kernel_size

        dynamic_out = fn.conv2d(
            x,
            weights,
            bias=bias,
            stride=stride,
            padding=1,
            dilation=1,
        )

        x = dynamic_out.flatten(2).transpose(1, 2)

        return x, waves


class DOFA(nn.Module):
    """
    Dynamic One-For-All (DOFA) v2 Encoder with improved architecture.

    This version uses patch size 14 and integrates with timm's VisionTransformer
    for better performance and standardization.

    References:
    * https://github.com/zhu-xlab/DOFA
    * https://github.com/microsoft/torchgeo/blob/main/torchgeo/models/dofa.py

    """

    def __init__(  # noqa: PLR0913
        self,
        encoder_name: str = "dofa_base",
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 14,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        interpolate_mode: str = "bicubic",
        out_layers: int | list[int] = -1,
        *,
        pretrained: bool = True,
        qkv_bias: bool = True,
        init_values: float = 1e-5,
        dynamic_img_size: bool = True,
        final_norm: bool = True,
        convert_patch_14_to_16: bool = False,
    ) -> None:
        """Initialize DOFAv2 Encoder."""
        super().__init__()

        self.encoder_name = encoder_name
        self.pretrained = pretrained

        # Handle image size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rate = attn_drop_rate
        self.final_norm = final_norm
        self.interpolate_mode = interpolate_mode
        self.dynamic_img_size = dynamic_img_size
        self.convert_patch_14_to_16 = convert_patch_14_to_16

        # Parse output layers
        if isinstance(out_layers, int):
            if out_layers == -1:
                out_layers = depth - 1
            self.out_layers = [out_layers]
        elif isinstance(out_layers, (list, tuple)):
            self.out_layers = list(out_layers)
        else:
            msg = "out_layers must be type of int, list or tuple"
            raise TypeError(msg)

        # Adjust output layers based on model size
        if encoder_name == "dofa_large" and self.out_layers == [
            11,
        ]:  # Default last layer
            self.out_layers = [21]  # v2 large has different indices

        # Create patch embedding layer
        self.patch_embed = DOFAEmbedding(
            wv_planes=128,
            dynamic_embed_dim=128,
            kernel_size=14,
            embed_dim=self.embed_dim,
            convert_patch_14_to_16=self.convert_patch_14_to_16,
        )

        # Calculate number of patches
        effective_patch_size = 16 if convert_patch_14_to_16 else 14
        self.num_patches = (self.img_size[0] // effective_patch_size) ** 2

        # Create VisionTransformer model from timm
        model_args = {
            "img_size": self.img_size,
            "patch_size": effective_patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "init_values": init_values,
            "num_classes": 0,
            "dynamic_img_size": self.dynamic_img_size,
        }
        self.model = VisionTransformer(**model_args)

        # Remove the default patch embedding projection
        del self.model.patch_embed.proj

        # Final normalization layer
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        # Initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize or load pretrained weights."""
        if self.pretrained:
            if self.encoder_name == "dofa_base":
                model_dict = dofa_encoder_base()
            elif self.encoder_name == "dofa_large":
                model_dict = dofa_encoder_large()
            else:
                msg = f"Unknown encoder name: {self.encoder_name}"
                raise ValueError(msg)

            # Handle position embedding size mismatch
            if (
                self.img_size != (224, 224)
                and "model.pos_embed" in model_dict
                and self.model.pos_embed.shape != model_dict["model.pos_embed"].shape
            ):
                h, w = self.img_size
                effective_patch_size = 16 if self.convert_patch_14_to_16 else 14
                pos_size = int(math.sqrt(model_dict["model.pos_embed"].shape[1] - 1))
                model_dict["model.pos_embed"] = self._resize_pos_embed(
                    model_dict["model.pos_embed"],
                    (h // effective_patch_size, w // effective_patch_size),
                    (pos_size, pos_size),
                    self.interpolate_mode,
                )

            # Load state dict
            missing_keys, unexpected_keys = self.load_state_dict(
                model_dict,
                strict=True,
            )

            # Allow certain missing keys for encoder-only usage
            allowed_missing = {
                "norm.weight",
                "norm.bias",
                "fc_norm.weight",
                "fc_norm.bias",
                "head.weight",
                "head.bias",
            }
            actual_missing = set(missing_keys) - allowed_missing

            if actual_missing:
                msg = f"Missing required keys in state dict: {actual_missing}"
                raise RuntimeError(msg)
            if unexpected_keys:
                msg = f"Unexpected keys in state dict: {unexpected_keys}"
                raise RuntimeError(msg)

    @staticmethod
    def _resize_pos_embed(
        pos_embed: Tensor,
        input_shape: tuple[int, int],
        pos_shape: tuple[int, int],
        mode: str,
    ) -> Tensor:
        """
        Resize position embeddings.

        Args:
            pos_embed: Position embedding weights [B, L, C].
            input_shape: Target shape (H, W).
            pos_shape: Original shape (H, W).
            mode: Interpolation mode.

        Returns:
            Resized position embeddings.

        """
        expected_ndim = 3
        if pos_embed.ndim != expected_ndim:
            msg = f"pos_embed must have shape [B, L, C], but got {pos_embed.ndim}"
            raise ValueError(msg)

        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]

        pos_embed_weight = pos_embed_weight.reshape(
            1,
            pos_h,
            pos_w,
            pos_embed.shape[2],
        ).permute(0, 3, 1, 2)

        pos_embed_weight = fn.interpolate(
            pos_embed_weight,
            size=input_shape,
            mode=mode,
            align_corners=False,
        )

        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)

        return torch.cat((cls_token_weight, pos_embed_weight), dim=1)

    def forward_features(self, x: Tensor, wavelengths: Tensor) -> list[Tensor]:
        """
        Forward pass through the encoder, extracting features.

        Args:
            x: Input image tensor of shape [B, C, H, W].
            wavelengths: Wavelengths of each spectral band (μm).

        Returns:
            List of feature maps from specified layers.

        """
        expected_ndim = 2
        if wavelengths.dim() == expected_ndim:
            # If all samples in batch have same wavelengths, use first sample
            if torch.allclose(wavelengths, wavelengths[0:1].expand_as(wavelengths)):
                wavelengths = wavelengths[0]  # Shape: [C]
            else:
                msg = "DOFA cannot handle different wavelengths within a batch"
                raise ValueError(msg)

        # Embed patches using dynamic convolution
        x, _ = self.patch_embed(x, wavelengths)

        batch_size, length, channels = x.shape
        hw = int(math.sqrt(length))

        # Reshape for dynamic image size if enabled
        if self.dynamic_img_size:
            x = x.view(batch_size, hw, hw, channels)

        # Add position embeddings
        x = self.model._pos_embed(x)  # noqa: SLF001

        # Apply patch dropout if in training mode
        x = self.model.patch_drop(x)

        # Pre-normalization if exists
        x = self.model.norm_pre(x)

        # Collect output features
        out_features = []

        # Apply transformer blocks
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.out_layers:
                # Extract features without class token
                feat = x[:, 1:] if self.model.has_class_token else x

                # Reshape to spatial format
                batch_size, length, channels = feat.shape
                feat = (
                    feat.reshape(batch_size, hw, hw, channels)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                out_features.append(feat)

        # Apply final normalization
        if self.final_norm:
            x = self.model.norm(x)

            # If last layer is requested and not already added
            if (self.depth - 1) in self.out_layers and len(out_features) < len(
                self.out_layers,
            ):
                feat = x[:, 1:] if self.model.has_class_token else x
                batch_size, length, channels = feat.shape
                feat = (
                    feat.reshape(batch_size, hw, hw, channels)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                out_features.append(feat)

        return out_features

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


# Factory functions for easy model creation
def create_dofa_base(
    img_size: int | tuple[int, int] = 224,
    out_layers: int | list[int] | None = None,
    *,
    pretrained: bool = True,
    **kwargs: object,
) -> DOFA:
    """
    Create DOFA base model.

    Args:
        img_size: Input image size.
        out_layers: Layers to extract features from.
        pretrained: Whether to load pretrained weights.
        **kwargs: Additional arguments for DOFA.

    Returns:
        DOFA base model instance.

    """
    return DOFA(
        encoder_name="dofa_base",
        img_size=img_size,
        patch_size=14,
        embed_dim=768,
        num_heads=12,
        depth=12,
        out_layers=out_layers or [4, 6, 10, 11],
        pretrained=pretrained,
        **kwargs,
    )


def create_dofa_large(
    img_size: int | tuple[int, int] = 224,
    out_layers: int | list[int] | None = None,
    *,
    pretrained: bool = True,
    **kwargs: object,
) -> DOFA:
    """
    Create DOFA large model.

    Args:
        img_size: Input image size.
        out_layers: Layers to extract features from.
        pretrained: Whether to load pretrained weights.
        **kwargs: Additional arguments for DOFA.

    Returns:
        DOFA large model instance.

    """
    return DOFA(
        encoder_name="dofa_large",
        img_size=img_size,
        patch_size=14,
        embed_dim=1024,
        num_heads=16,
        depth=24,
        out_layers=out_layers or [5, 9, 15, 21],
        pretrained=pretrained,
        **kwargs,
    )
