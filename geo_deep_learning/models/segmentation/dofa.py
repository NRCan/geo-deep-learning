import torch
import torch.nn.functional as F
from models.encoders.dofa import DOFA
from models.necks.multilevel_neck import MultiLevelNeck
from models.decoders.upernet import UperNetDecoder
from models.heads.segmentation_head import SegmentationHead, SegmentationOutput
from models.heads.fcn_head import FCNHead
from .base import BaseSegmentationModel
from typing import Any


class DOFASegmentationModel(BaseSegmentationModel):
    def __init__(self, 
                 encoder: str = "dofa_base", 
                 pretrained: bool = True, 
                 image_size: tuple[int, int] = (512, 512), 
                 wavelengths: list[float] = [0.665, 0.549, 0.481],
                 freeze_layers: list[str] = None,
                 num_classes: int = 1,
                 *args: Any, **kwargs: Any):
        super().__init__(DOFA, MultiLevelNeck, UperNetDecoder, SegmentationHead, SegmentationOutput)
        
        if encoder == "dofa_base":
            self.embed_dim = 768
            kwargs |= {'patch_size': 16, 
                       'embed_dim': self.embed_dim, 
                       'depth': 12, 
                       'num_heads': 12, 
                       'out_layers': [3, 6, 9, 11]}
        
        elif encoder == "dofa_large":
            self.embed_dim = 1024
            kwargs |= {'patch_size': 16, 
                       'embed_dim': self.embed_dim, 
                       'depth': 24, 
                       'num_heads': 16, 
                       'out_layers': [5, 11, 17, 23]}
        
        self.encoder = DOFA(encoder_name=encoder, 
                            pretrained=pretrained,
                            img_size=image_size, 
                            wavelengths=wavelengths, 
                            *args, **kwargs)
        
        self.neck = MultiLevelNeck(in_channels=[self.embed_dim] * 4,
                                   out_channels=[self.embed_dim] * 4,
                                   scales=[4, 2, 1, 0.5])
        self.decoder = UperNetDecoder(embed_dim=[self.embed_dim] * 4,
                                      pool_scales=(1, 2, 3, 6),
                                      channels=256,
                                      align_corners=False,
                                      scale_modules=False)
        self.aux_head = FCNHead(in_channels=self.embed_dim, 
                                channels=256, 
                                num_convs=1, 
                                num_classes=num_classes)
        
        self.head = SegmentationHead(in_channels=256, num_classes=num_classes)
        self.output_struct = SegmentationOutput
        
        if freeze_layers:
            self._freeze_layers(layers=freeze_layers)
    
    def forward(self, x):
        image_size = x.shape[2:]
        x = self.encoder(x)
        feats = self.neck(x)
        x = self.decoder(feats)
        x = self.head(x)
        x = F.interpolate(input=x, size=image_size, 
                          scale_factor=None, mode='bilinear', align_corners=False)
        
        aux_x = self.aux_head(feats[-1])
        aux_x = F.interpolate(input=aux_x, size=image_size, 
                              scale_factor=None, mode='bilinear', align_corners=False)
        
        return self.output_struct(out=x, aux=aux_x)

if __name__ == "__main__":
    model = DOFASegmentationModel()
    x = torch.randn(5, 3, 512, 512)
    outputs = model(x)
    print(f"outputs.shape: {outputs.out.shape}")
    print(f"aux_outputs.shape: {outputs.aux.shape}")
