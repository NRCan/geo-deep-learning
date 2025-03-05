import torch.nn as nn

class BaseSegmentationModel(nn.Module):
    def __init__(self, encoder, neck, decoder, head, auxilary_head=None):
        super().__init__()
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.auxilary_head = auxilary_head
        self.head = head
        
    def forward(self, x):
        outputs = {}
        x = self.encoder(x)
        x = self.neck(x)
        x = self.decoder(x)
        if self.auxilary_head:
            aux = self.auxilary_head(x)
            outputs['aux'] = aux
        x = self.head(x)
        outputs['out'] = x
        return outputs
    
    def _freeze_layers(self, layers):
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = False
