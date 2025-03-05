import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSegmentationModel
from models.encoders.mix_transformer import get_encoder
from models.decoders.segformer_mlp import Decoder

class SegFormerSegmentationModel(BaseSegmentationModel):
    def __init__(self, 
                 encoder: str = "mit_b0", 
                 in_channels: int = 3, 
                 weights: str = None, 
                 freeze_layers: list[str] = None, 
                 num_classes: int = 1) -> None:
        super().__init__()
        self.encoder = get_encoder(name=encoder, 
                                   in_channels=in_channels,
                                   depth=5, weights=weights, 
                                   drop_path_rate=0.1)
        if freeze_layers:
            self._freeze_layers(layers=freeze_layers)
        
        self.decoder = Decoder(encoder=encoder, num_classes=num_classes)

    def forward(self, img):
        # print(f"{__name__}: Input shape: {img.shape}")
        x = self.encoder(img)[2:]
        x = self.decoder(x)
        x = F.interpolate(input=x, size=img.shape[2:], scale_factor=None, mode='bilinear', align_corners=False)
        return x
    
if __name__ == "__main__":
    model = SegFormerSegmentationModel()
    x = torch.randn(5, 3, 512, 512)
    outputs = model(x)
    print(f"outputs.shape: {outputs.shape}")