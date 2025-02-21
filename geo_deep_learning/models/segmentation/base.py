import torch.nn as nn

class BaseSegmentationModel(nn.Module):
    def __init__(self, encoder, neck, decoder, head, output_struct, auxilary_head=None):
        super().__init__()
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.auxilary_head = auxilary_head
        self.head = head
        self.output_struct = output_struct
    def forward(self, x):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.decoder(x)
        aux = None
        if self.auxilary_head:
            aux = self.auxilary_head(x)
        x = self.head(x)
        output = self.output_struct(out=x, aux=aux)
        return output
    
    def _freeze_layers(self, layers):
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = False
