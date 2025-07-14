"""DOFA segmentation model."""

import torch
import torch.nn.functional as fn
from models.decoders.upernet import UperNetDecoder
from models.encoders.dofa import DOFA, create_dofa_base, create_dofa_large
from models.heads.fcn_head import FCNHead
from models.heads.segmentation_head import SegmentationHead, SegmentationOutput
from models.necks.multilevel_neck import MultiLevelNeck
from torch import Tensor

from .base import BaseSegmentationModel


class DOFASegmentationModel(BaseSegmentationModel):
    """DOFA segmentation model."""

    def __init__(
        self,
        encoder: str = "dofa_base",
        image_size: tuple[int, int] = (512, 512),
        freeze_layers: list[str] | None = None,
        num_classes: int = 1,
        *,
        pretrained: bool = True,
    ) -> None:
        """Initialize DOFA segmentation model."""
        super().__init__(
            DOFA,
            MultiLevelNeck,
            UperNetDecoder,
            SegmentationHead,
            SegmentationOutput,
        )

        if encoder == "dofa_base":
            self.encoder = create_dofa_base(img_size=image_size, pretrained=pretrained)

        elif encoder == "dofa_large":
            self.encoder = create_dofa_large(img_size=image_size, pretrained=pretrained)

        self.neck = MultiLevelNeck(
            in_channels=[self.embed_dim] * 4,
            out_channels=[self.embed_dim] * 4,
            scales=[4, 2, 1, 0.5],
        )
        self.decoder = UperNetDecoder(
            embed_dim=[self.embed_dim] * 4,
            pool_scales=(1, 2, 3, 6),
            channels=256,
            align_corners=False,
            scale_modules=False,
        )
        self.aux_head = FCNHead(
            in_channels=self.embed_dim,
            channels=256,
            num_convs=1,
            num_classes=num_classes,
        )

        self.head = SegmentationHead(in_channels=256, num_classes=num_classes)
        self.output_struct = SegmentationOutput

        if freeze_layers:
            self._freeze_layers(layers=freeze_layers)

    def forward(self, x: Tensor) -> SegmentationOutput:
        """Forward pass."""
        image_size = x.shape[2:]
        x = self.encoder(x)
        feats = self.neck(x)
        x = self.decoder(feats)
        x = self.head(x)
        x = fn.interpolate(
            input=x,
            size=image_size,
            scale_factor=None,
            mode="bilinear",
            align_corners=False,
        )

        aux_x = self.aux_head(feats[-1])
        aux_x = fn.interpolate(
            input=aux_x,
            size=image_size,
            scale_factor=None,
            mode="bilinear",
            align_corners=False,
        )

        return self.output_struct(out=x, aux=aux_x)


if __name__ == "__main__":
    model = DOFASegmentationModel()
    x = torch.randn(5, 3, 512, 512)
    outputs = model(x)
    # print(f"outputs.shape: {outputs.out.shape}")
    # print(f"aux_outputs.shape: {outputs.aux.shape}")
