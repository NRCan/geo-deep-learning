
from torch import nn
from models.hrnet.utils import ModelHelpers
from models.hrnet.ocr_modules import SpatialGather_Module, SpatialOCR_Module


BNReLU = ModelHelpers.BNReLU

class OCR(nn.Module):
    
    def __init__(self, num_classes, high_level_ch) -> None:
        super(OCR, self).__init__()
        
        ocr_mid_channels = 512
        ocr_key_channels = 256
        
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels),)  
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        
        self.cls_head = nn.Conv2d(ocr_mid_channels, num_classes, 
                                  kernel_size=1, stride=1, padding=0,bias=True)

        self.aux_head = nn.Sequential(nn.Conv2d(high_level_ch, high_level_ch, 
                                                kernel_size=1, stride=1, padding=0),
                                      BNReLU(high_level_ch),
                                      nn.Conv2d(high_level_ch, num_classes,
                                                kernel_size=1, stride=1, padding=0, bias=True))
        
    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats
        
    