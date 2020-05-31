import torch
import numpy as np
import copy
import torchvision
from torch import nn
import torchvision.models as models
from models import common


class LayerExtractor(nn.Module):
    def __init__(self, submodule, extracted_layer, leftover=False):
        super(LayerExtractor, self).__init__()
        # TODO: documentation
        self.submodule = submodule
        self.extracted_layer = extracted_layer
        self.leftover_out = leftover
        # Extract the output size of the layer to fit resize after the concatenation TODO put it in doc
        self.out_channels = self.submodule.backbone.conv1.out_channels

    def forward(self, x):
        if self.extracted_layer == 'conv1': 
            # Extract all layers in the ResNet backbone and [:1] for the conv1
            modules = list(self.submodule.backbone.children())[:1]
            modules = nn.Sequential(*modules)
            # Extract all the other layers following the layer of extraction
            leftover = list(self.submodule.backbone.children())[1:]
            classifier = list(self.submodule.classifier.children())
            leftover.extend(classifier)

        # TODO: change the rest to fit the others entries
        elif self.extracted_layer == 'inner-layer-3':                     
            modules = list(self.submodule.children())[:6]
            third_module = list(self.submodule.children())[6]
            third_module_modules = list(third_module.children())[:3]    # take the first three inner modules
            third_module = nn.Sequential(*third_module_modules)
            modules.append(third_module)
        elif self.extracted_layer == 'layer-3':                     
            modules = list(self.submodule.children())[:7]
        else:                                               # after avg-pool
            modules = list(self.submodule.children())[:9]

        # Return the rest of the Network or only the first part
        if self.leftover_out:
            modules_layers = leftover
        else:
            modules_layers = modules

        self.submodule = nn.Sequential(*modules_layers)
        x = self.submodule(x)
        return x


class MyEnsemble(nn.Module):
    def __init__(self, num_channels):
        super(MyEnsemble, self).__init__()
        # TODO: documentation
        model_rgb = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, aux_loss=None)
        model_rgb.classifier = common.DeepLabHead(2048, num_channels)
        self.modelRGB = LayerExtractor(model_rgb, 'conv1')

        model_nir = copy.deepcopy(model_rgb)
        model_nir.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.modelNIR = LayerExtractor(model_nir, 'conv1')

        self.leftover = LayerExtractor(model_rgb, 'conv1', leftover=True)

        self.conv1x1 = nn.Conv2d(
                in_channels=self.modelRGB.out_channels*2,
                out_channels=self.modelRGB.out_channels,
                kernel_size=1
        )

    def forward(self, x1, x2):
        rgb = self.modelRGB(x1)
        nir = self.modelNIR(x2)
   
        #print('shape de rgb apres', rgb.shape)
        #print('shape de nir apres', nir.shape)
        
        # TODO: concatenation
        x = torch.cat((rgb, nir), dim=1)

        #print('shape of concatenation', x.shape)

        # TODO: conv 1x1 need to match the enter of the bn1
        x = self.conv1x1(x)

        #print('shape after conv 1x1', x.shape)

        # TODO: give the result to the reste of the network
        x = self.leftover(x)
        
        #print('shape after the rest of the network', x.shape)

        return x

