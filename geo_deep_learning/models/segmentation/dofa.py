"""DOFA segmentation model."""

import kornia as krn
import torch
import torch.nn.functional as fn
from kornia.augmentation import AugmentationSequential

from geo_deep_learning.models.decoders.upernet import UperNetDecoder
from geo_deep_learning.models.encoders.dofa_v2 import (
    DOFAv2,
    create_dofa_base,
    create_dofa_large,
)
from geo_deep_learning.models.heads.fcn_head import FCNHead
from geo_deep_learning.models.heads.segmentation_head import (
    SegmentationHead,
    SegmentationOutput,
)
from geo_deep_learning.models.necks.multilevel_neck import MultiLevelNeck

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
            DOFAv2,
            MultiLevelNeck,
            UperNetDecoder,
            SegmentationHead,
            SegmentationOutput,
        )

        if encoder == "dofa_base":
            self.embed_dim = 768
            self.encoder = create_dofa_base(img_size=image_size, pretrained=pretrained)

        elif encoder == "dofa_large":
            self.embed_dim = 1024
            self.encoder = create_dofa_large(img_size=image_size, pretrained=pretrained)
        else:
            msg = f"Invalid encoder: {encoder}"
            raise ValueError(msg)

        self.neck = MultiLevelNeck(
            in_channels=[self.embed_dim] * 4,
            out_channels=[self.embed_dim] * 4,
            scales=[4, 2, 1, 0.5],
            norm_cfg={"type": "BN"},
            act_cfg={"type": "ReLU"},
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

    def forward(self, x: torch.Tensor, wavelengths: torch.Tensor) -> SegmentationOutput:
        """Forward pass."""
        image_size = x.shape[2:]
        x = self.encoder(x, wavelengths)
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


class DataAugmentation(torch.nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, patch_size: tuple[int, int]) -> None:
        """Initialize data augmentation module."""
        super().__init__()
        self.patch_size = patch_size

        random_resized_crop_zoom_in = krn.augmentation.RandomResizedCrop(
            size=self.patch_size,
            scale=(1.0, 2.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )
        random_resized_crop_zoom_out = krn.augmentation.RandomResizedCrop(
            size=self.patch_size,
            scale=(0.5, 1.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )

        self.transforms = AugmentationSequential(
            krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomRotation90(
                times=(1, 3),
                p=0.5,
                align_corners=True,
                keepdim=True,
            ),
            random_resized_crop_zoom_in,
            random_resized_crop_zoom_out,
            data_keys=None,
            random_apply=1,
        )

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply augmentations to batch.

        Args:
            batch: Dict with keys 'image' and 'mask' (B, C, H, W)

        Returns:
            Dict with augmented 'image' and 'mask' tensors

        """
        transformed = self.transforms(
            {"image": batch["image"], "mask": batch["mask"]},
        )
        batch.update(
            {
                "image": transformed["image"],
                "mask": transformed["mask"],
            },
        )

        return batch


if __name__ == "__main__":
    model = DOFASegmentationModel()
    x = torch.randn(5, 3, 512, 512)
    wavelengths = torch.tensor([0.665, 0.549, 0.481])
    outputs = model(x, wavelengths)
    # print(f"outputs.shape: {outputs.out.shape}")
    # print(f"aux_outputs.shape: {outputs.aux.shape}")
