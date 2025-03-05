import torch.nn as nn
from models.utils import patch_first_conv   


class BaseSegmentationModel(nn.Module):
    def __init__(self, 
                 encoder=None, 
                 neck=None, 
                 decoder=None, 
                 head=None, 
                 output_struct=None, 
                 auxilary_head=None):
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

class EncoderMixin:
    # Taken from segmentation models pytorch
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _output_stride = 32

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)
