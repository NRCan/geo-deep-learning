"""SegFormer segmentation model."""

import torch
import torch.nn.functional as fn
from models.decoders.segformer_mlp import Decoder
from models.encoders.mix_transformer import get_encoder

from .base import BaseSegmentationModel


class SegFormerSegmentationModel(BaseSegmentationModel):
    """SegFormer segmentation model."""

    def __init__(
        self,
        encoder: str = "mit_b0",
        in_channels: int = 3,
        weights: str | None = None,
        freeze_layers: list[str] | None = None,
        num_classes: int = 1,
    ) -> None:
        """Initialize SegFormer segmentation model."""
        super().__init__()
        self.encoder = get_encoder(
            name=encoder,
            in_channels=in_channels,
            depth=5,
            weights=weights,
            drop_path_rate=0.1,
        )
        if freeze_layers:
            self._freeze_layers(layers=freeze_layers)

        self.decoder = Decoder(encoder=encoder, num_classes=num_classes)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.encoder(img)[2:]
        x = self.decoder(x)
        return fn.interpolate(
            input=x,
            size=img.shape[2:],
            scale_factor=None,
            mode="bilinear",
            align_corners=False,
        )


if __name__ == "__main__":
    model = SegFormerSegmentationModel()
    x = torch.randn(5, 3, 512, 512)
    outputs = model(x)
    # print(f"outputs.shape: {outputs.shape}")
