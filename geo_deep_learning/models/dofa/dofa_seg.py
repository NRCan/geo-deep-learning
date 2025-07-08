"""Dynamic One-For-All (DOFA) models for segmentation."""

import math
import warnings
from functools import partial

import torch
import torch.nn.functional as fn
from neckhead import MultiLevelNeck
from timm.models.vision_transformer import Block
from torch import Tensor, nn
from torch.nn import init


def resize(  # noqa: PLR0913
    input_tensor: Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "nearest",
    *,
    align_corners: bool | None = None,
    warning: bool = True,
) -> Tensor:
    """Resize a tensor."""
    if warning and size is not None and align_corners:
        input_h, input_w = tuple(int(x) for x in input_tensor.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if (output_h > input_h or output_w > output_h) and (
            (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
            and (output_h - 1) % (input_h - 1)
            and (output_w - 1) % (input_w - 1)
        ):
            warnings.warn(
                f"When align_corners={align_corners}, "
                "the output would more aligned if "
                f"input size {(input_h, input_w)} is `x+1` and "
                f"out size {(output_h, output_w)} is `nx+1`",
                stacklevel=2,
            )

    return fn.interpolate(input_tensor, size, scale_factor, mode, align_corners)


def position_embedding(embed_dim: int, pos: Tensor) -> Tensor:
    """
    Compute the 1D sine/cosine position embedding.

    Args:
        embed_dim: Output dimension D for each position. Must be even.
        pos: A list of positions to be encoded, of size (M,).

    Returns:
        Position embeddings of size (M, D).

    Raises:
        AssertionError: If *embed_dim* is not even.

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
            num_heads: Number of heads.
            num_layers: Number of layers.

        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input mini-batch of size (seq_len, batch, input_dim).

        Returns:
            Weight and bias.

        """
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        # Using the last output to generate bias
        bias = self.fc_bias(transformer_output[-1])
        return weights, bias


class FCResLayer(nn.Module):
    """Fully-connected residual layer."""

    def __init__(self, linear_size: int = 128) -> None:
        """
        Initialize a new FCResLayer instance.

        Args:
            linear_size: Size of linear layer.

        """
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input mini-batch.

        Returns:
            Output of the model.

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
        dynamic_embed_dim: int,
        kernel_size: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        """
        Initialize a new DOFAEmbedding instance.

        Args:
            dynamic_embed_dim: Dimensions of dynamic weight generator.
            kernel_size: Kernel size of the depth-wise convolution.
            embed_dim: Embedding dimensions.

        """
        super().__init__()
        self.dynamic_embed_dim = dynamic_embed_dim
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            dynamic_embed_dim,
            self._num_kernel,
            embed_dim,
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(dynamic_embed_dim)

        self._init_weights()

    def _init_weight(self, m: object) -> None:
        """
        Initialize weights of a single layer.

        Args:
            m: A single layer.

        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        """Initialize weights of all layers."""
        self.weight_generator.apply(self._init_weight)
        self.fclayer.apply(self._init_weight)

    def forward(self, x: Tensor, wavelengths: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input mini-batch.
            wavelengths: Wavelengths of each spectral band (μm).

        Return:
            Output mini-batch and wavelengths.

        """
        inplanes = wavelengths.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = position_embedding(self.dynamic_embed_dim, wavelengths * 1000)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)  # 3x3x3

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

        dynamic_out = fn.conv2d(
            x,
            weights,
            bias=bias,
            stride=self.kernel_size,
            padding=1,
            dilation=1,
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class Encoder(nn.Module):
    """
    Dynamic One-For-All (DOFA) model for segmentation.

    References:
    * https://github.com/zhu-xlab/DOFA
    * https://github.com/microsoft/torchgeo/blob/main/torchgeo/models/dofa.py

    """

    def __init__(  # noqa: PLR0913
        self,
        encoder_name: str = "dofa_base",
        img_size: tuple = (224, 224),
        wavelengths: list[float] | None = None,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        interpolate_mode: str = "bicubic",
        out_layers: int | list[int] = -1,
        *,
        pretrained: bool = True,
        qkv_bias: bool = True,
        pre_norm: bool = False,
        final_norm: bool = False,
    ) -> None:
        """Initialize Encoder."""
        self.encoder_name = encoder_name
        self.pretrained = pretrained
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
        self.wavelengths = wavelengths or [0.665, 0.549, 0.481]
        self.pre_norm = pre_norm
        self.final_norm = final_norm
        self.out_layers = out_layers
        self.interpolate_mode = interpolate_mode
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if isinstance(out_layers, int):
            if out_layers == -1:
                out_layers = depth - 1
            self.out_layers = [out_layers]
        elif isinstance(out_layers, (list, tuple)):
            self.out_layers = out_layers
        else:
            msg = "out_indices must be type of int, list or tuple"
            raise TypeError(msg)

        super().__init__()

        self.patch_embed = DOFAEmbedding(
            dynamic_embed_dim=128,
            kernel_size=16,
            embed_dim=self.embed_dim,
        )
        self.num_patches = (self.img_size[0] // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim),
        )
        self.drop_after_pos = nn.Dropout(p=self.drop_rate)

        self.blocks = nn.ModuleList(
            [
                Block(
                    self.embed_dim,
                    self.num_heads,
                    self.mlp_ratio,
                    self.qkv_bias,
                    norm_layer=self.norm_layer,
                )
                for i in range(self.depth)
            ],
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights of the model."""
        if self.pretrained:
            # print(f"Loading pretrained weights for {self.encoder_name}")
            if self.encoder_name == "dofa_base":
                model_dict = dofa_encoder_base()
            elif self.encoder_name == "dofa_large":
                model_dict = dofa_encoder_large()

            if (
                "pos_embed" in model_dict
                and self.pos_embed.shape != model_dict["pos_embed"].shape
            ):
                h, w = self.img_size
                pos_size = int(math.sqrt(model_dict["pos_embed"].shape[1] - 1))
                model_dict["pos_embed"] = self.resize_pos_embed(
                    model_dict["pos_embed"],
                    (h // self.patch_size, w // self.patch_size),
                    (pos_size, pos_size),
                    self.interpolate_mode,
                )
            missing_keys, unexpected_keys = self.load_state_dict(
                model_dict,
                strict=False,
            )
            if missing_keys:
                msg = f"Missing keys in state dict: {missing_keys}"
                raise RuntimeError(msg)
            if unexpected_keys:
                msg = f"Unexpected keys in state dict: {unexpected_keys}"
                raise RuntimeError(msg)

    def _pos_embeding(
        self,
        patched_img: Tensor,
        hw_shape: tuple[int, int],
        pos_embed: Tensor,
    ) -> Tensor:
        """
        Positioning embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.

        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].

        Return:
            torch.Tensor: The pos encoded image feature.

        """
        expected_ndim = 3
        if patched_img.ndim != expected_ndim or pos_embed.ndim != expected_ndim:
            msg = "the shapes of patched_img and pos_embed must be [B, L, C]"
            raise ValueError(msg)
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if (
                pos_len
                == (self.img_size[0] // self.patch_size)
                * (self.img_size[1] // self.patch_size)
                + 1
            ):
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                msg = f"Unexpected shape of pos_embed, got {pos_embed.shape}."
                raise ValueError(msg)
            pos_embed = self.resize_pos_embed(
                pos_embed,
                hw_shape,
                (pos_h, pos_w),
                self.interpolate_mode,
            )
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(
        pos_embed: Tensor,
        input_shpae: tuple[int, int],
        pos_shape: tuple[int, int],
        mode: str,
    ) -> Tensor:
        """
        Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.

        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]

        """
        expected_ndim = 3
        if pos_embed.ndim != expected_ndim:
            msg = "shape of pos_embed must be [B, L, C]"
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
        pos_embed_weight = resize(
            pos_embed_weight,
            size=input_shpae,
            align_corners=False,
            mode=mode,
        )
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return torch.cat((cls_token_weight, pos_embed_weight), dim=1)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass of the model."""
        batch_size = x.shape[0]
        if self.wavelengths is None:
            msg = "Wavelengths must be provided"
            raise ValueError(msg)
        wavelist = torch.tensor(self.wavelengths, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)
        hw = self.img_size[0] // self.patch_embed.kernel_size
        hw_shape = (hw, hw)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)
        partial_norm = partial(nn.LayerNorm, eps=1e-6)
        pre_norm = partial_norm(self.embed_dim)
        final_norm = partial_norm(self.embed_dim)

        if self.pre_norm:
            x = pre_norm(x)
        outs = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1 and self.final_norm:
                x = final_norm(x)
            if i in self.out_layers:
                out = x[:, 1:]
                batch_size, _, channels = out.shape
                out = (
                    out.reshape(batch_size, hw_shape[0], hw_shape[1], channels)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)

        return outs


def dofa_encoder_base() -> dict[str, Tensor]:
    """Load DOFA encoder base model."""
    url: str = "https://huggingface.co/XShadow/DOFA/resolve/main/DOFA_ViT_base_e100.pth"
    model_dict = torch.hub.load_state_dict_from_url(
        url,
        progress=True,
        map_location="cpu",
    )
    del model_dict["mask_token"]
    del model_dict["norm.weight"], model_dict["norm.bias"]
    del model_dict["projector.weight"], model_dict["projector.bias"]
    return model_dict


def dofa_encoder_large() -> dict[str, Tensor]:
    """Load DOFA encoder large model."""
    url: str = (
        "https://huggingface.co/XShadow/DOFA/resolve/main/DOFA_ViT_large_e100.pth"
    )
    model_dict = torch.hub.load_state_dict_from_url(
        url,
        progress=True,
        map_location="cpu",
    )
    del model_dict["mask_token"]
    del model_dict["norm.weight"], model_dict["norm.bias"]
    del model_dict["projector.weight"], model_dict["projector.bias"]
    return model_dict


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768) -> None:
        """Initialize MLP."""
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = x.flatten(2).transpose(1, 2).contiguous()
        return self.proj(x)


class Decoder(nn.Module):
    """Decoder for DOFA segmentation."""

    def __init__(
        self,
        in_channels: list[int] | None = None,
        embedding_dim: int = 768,
        num_classes: int = 1,
        dropout_ratio: float = 0.1,
    ) -> None:
        """Initialize Decoder."""
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        if in_channels is None:
            in_channels = [768, 768, 768, 768]
        expected_in_channels = 4
        if len(in_channels) != expected_in_channels:
            msg = "in_channels must be a list of 4 integers"
            raise ValueError(msg)
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = (
            self.in_channels
        )

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dim * 4,
                out_channels=embedding_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x: list[Tensor]) -> Tensor:
        """Forward pass."""
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape

        _c4 = (
            self.linear_c4(c4)
            .permute(0, 2, 1)
            .reshape(n, -1, c4.shape[2], c4.shape[3])
            .contiguous()
        )
        _c4 = fn.interpolate(
            input=_c4,
            size=c1.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        _c3 = (
            self.linear_c3(c3)
            .permute(0, 2, 1)
            .reshape(n, -1, c3.shape[2], c3.shape[3])
            .contiguous()
        )
        _c3 = fn.interpolate(
            input=_c3,
            size=c1.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        _c2 = (
            self.linear_c2(c2)
            .permute(0, 2, 1)
            .reshape(n, -1, c2.shape[2], c2.shape[3])
            .contiguous()
        )
        _c2 = fn.interpolate(
            input=_c2,
            size=c1.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        _c1 = (
            self.linear_c1(c1)
            .permute(0, 2, 1)
            .reshape(n, -1, c1.shape[2], c1.shape[3])
            .contiguous()
        )
        _c1 = fn.interpolate(
            input=_c1,
            size=c1.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        return self.linear_pred(x)


class DOFASeg(nn.Module):
    """DOFA segmentation model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str,
        image_size: tuple = (224, 224),
        wavelengths: list[float] | None = None,
        num_classes: int = 1,
        *,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        **kwargs: object,
    ) -> None:
        """Initialize DOFASeg."""
        super().__init__()
        if wavelengths is None:
            wavelengths = [0.665, 0.549, 0.481]
        if encoder == "dofa_base":
            kwargs |= {
                "patch_size": 16,
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "out_layers": [2, 5, 8, 11],
            }
            self.encoder = Encoder(
                encoder_name=encoder,
                pretrained=pretrained,
                img_size=image_size,
                wavelengths=wavelengths,
                **kwargs,
            )
            self.in_channels = [768, 768, 768, 768]
            self.embedding_dim = 768
        elif encoder == "dofa_large":
            kwargs |= {
                "patch_size": 16,
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "out_layers": [3, 7, 11, 23],
            }
            self.encoder = Encoder(
                encoder_name=encoder,
                pretrained=pretrained,
                img_size=image_size,
                wavelengths=wavelengths,
                **kwargs,
            )
            self.in_channels = [1024, 1024, 1024, 1024]
            self.embedding_dim = 1024
        else:
            msg = f"Unknown encoder: {encoder}"
            raise ValueError(msg)

        if freeze_encoder:
            self._freeze_encoder()
            self.encoder.eval()

        encoder_out_channels = [64, 128, 320, 512]
        self.neck = MultiLevelNeck(
            in_channels=self.in_channels,
            out_channels=encoder_out_channels,
            scales=[4, 2, 1, 0.5],
        )

        self.decoder = Decoder(
            in_channels=encoder_out_channels,
            embedding_dim=self.embedding_dim,
            num_classes=num_classes,
        )

    def _freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        image_size = x.shape[2:]
        encoder_features = self.encoder(x)
        neck_features = self.neck(encoder_features)
        decoder_features = self.decoder(neck_features)
        return fn.interpolate(
            input=decoder_features,
            size=image_size,
            scale_factor=None,
            mode="bilinear",
            align_corners=False,
        )


if __name__ == "__main__":
    batch_size = 8
    img = torch.rand(batch_size, 3, 512, 512)
    model = DOFASeg(
        encoder="dofa_base",
        pretrained=True,
        image_size=(512, 512),
        num_classes=5,
    )
    out = model(img)
    # print(f"Output shape: {out.shape}")
