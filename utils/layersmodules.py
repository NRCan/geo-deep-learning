import torch
import copy
from torch import nn
import torchvision.models as models
from collections import OrderedDict
from torch.nn import functional as F


class LayersEnsemble(nn.Module):
    def __init__(self, model, conc_point='conv1'):
        super(LayersEnsemble, self).__init__()
        # TODO: documentation

        # Init model
        model_rgb = model
        model_nir = copy.deepcopy(model)

        # Ajusting the second entry
        model_nir.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        ###################################################
        # TODO: blabla + if or a dict with fonctions
        self.modelRGB = model_rgb.backbone.conv1
        self.modelNIR = model_nir.backbone.conv1

        out_channels  = model_rgb.backbone.conv1.out_channels

        # TODO: Dict with name to deep number **dont forget the backbone
        self.leftover.backbone = nn.Sequential(
                *list(model_rgb.backbone.children())[1:]
        )

        # TODO: change for conc_point 9 after aspp
        self.classifier = model_rgb.classifier

        ####################################################

        # Conv Layer to fit the size of the next layer
        self.conv1x1 = nn.Conv2d(
                in_channels = out_channels*2, out_channels = out_channels, kernel_size=1
        )
        
        del model_nir, model_rgb

    def forward(self, x1, x2):
        input_shape = x1.shape[-2:]
        # Contract: features is a dict of tensors
        result = OrderedDict()

        rgb = self.modelRGB(x1)
        nir = self.modelNIR(x2)
        #print('shape de rgb apres', rgb.shape)
        #print('shape de nir apres', nir.shape)
        
        # Concatenation
        x = torch.cat((rgb, nir), dim=1)
        #print('shape of concatenation', x.shape)

        # Conv 1x1 need to match the enter of the next layer
        x = self.conv1x1(x)
        #print('shape after conv 1x1', x.shape)
        # Give the result to the rest of the network
        # TODO: something if deep 9
        x = self.leftover(x)

        # Give the result to the classifier
        x = self.classifier(x)
        #print('shape after classifier', x.shape)
        
        # See https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py
        # for more info 
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result['out'] = x

        return result

