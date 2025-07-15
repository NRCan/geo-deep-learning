"""Base segmentation model."""

from torch import Tensor, nn

from geo_deep_learning.models.utils import patch_first_conv


class BaseSegmentationModel(nn.Module):
    """Base segmentation model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: nn.Module | None = None,
        neck: nn.Module | None = None,
        decoder: nn.Module | None = None,
        head: nn.Module | None = None,
        output_struct: nn.Module | None = None,
        auxilary_head: nn.Module | None = None,
    ) -> None:
        """Initialize base segmentation model."""
        super().__init__()
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.auxilary_head = auxilary_head
        self.head = head
        self.output_struct = output_struct

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.encoder(x)
        x = self.neck(x)
        x = self.decoder(x)
        aux = None
        if self.auxilary_head:
            aux = self.auxilary_head(x)
        x = self.head(x)
        return self.output_struct(out=x, aux=aux)

    def _freeze_layers(self, layers: list[str]) -> None:
        """Freeze layers."""
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = False


class EncoderMixin:
    """Encoder mixin."""

    _output_stride = 32

    @property
    def out_channels(self) -> list[int]:
        """Return channels dimensions for each tensor of forward output of encoder."""
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self) -> int:
        """Return output stride."""
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels: int, *, pretrained: bool = True) -> None:
        """Change first convolution channels."""
        expected_in_channels = 3
        if in_channels == expected_in_channels:
            return

        self._in_channels = in_channels
        expected_out_channels = 3
        if self._out_channels[0] == expected_out_channels:
            self._out_channels = (in_channels, *self._out_channels[1:])

        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)
