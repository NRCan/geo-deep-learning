import logging
import torch.nn.functional as F

from torch import nn
from models.hrnet.ocr import OCR
from models.hrnet.backbone import hrnetv2



class HRNet(nn.Module):
    """High Resolution Network (hrnet_w48_v2) with Object Contextual Representation module

    Args:
        pretrained (bool): use pretrained weights 
        in_channels (int): number of bands/channels
        classes (int): number of classes
    """
    def __init__(self, pretrained, in_channels, classes) -> None:
        super(HRNet, self).__init__()
        if in_channels != 3:
            logging.critical(F"HRNet model expects three channels input")
        self.encoder = hrnetv2(num_of_classes=classes, pretrained=pretrained)
        high_level_ch = self.encoder.high_level_ch
        self.decoder = OCR(num_classes=classes, high_level_ch=high_level_ch)
    
    def forward(self, input):
        high_level_features = self.encoder(input)
        cls_out, aux_out, _ = self.decoder(high_level_features)
        
        input_size = input.shape[2:]
        aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
        cls_out = F.interpolate(cls_out, size=input_size, mode='bilinear', align_corners=False)
        if self.training:
            return cls_out, aux_out
        else:
            return cls_out

if __name__ == "__main__":
    import torch
    from torchinfo import summary
    
    model = HRNet(pretrained=True, in_channels=3, classes=4)
    model.to("cuda")
    batch_size = 4
    
    mask_tensor = torch.randn([batch_size, 3, 512, 512]).cuda()
    
    output, output_aux = model(mask_tensor)
    for name, para in model.named_parameters():
            print("-"*20)
            print(f"name: {name}")
            print(f"requires_grad: {para.requires_grad}")
    # print(output.shape)
    # print(output_aux.shape)
    # summary(model, input_size=(batch_size, 3, 512, 512))
    