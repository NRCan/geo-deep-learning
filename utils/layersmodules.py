import torch
import copy
from torch import nn
import torchvision.models as models
from collections import OrderedDict
from torch.nn import functional as F


nir_layers = {'conv1':1, 'maxpool':4, 'layer2':6, 'layer3':7, 'layer4':8}

class LayersEnsemble(nn.Module):
    def __init__(self, model, conc_point='conv1'):
        super(LayersEnsemble, self).__init__()
        # TODO: documentation

        # Init model
        model_rgb = model
        model_nir = copy.deepcopy(model)

        # Ajusting the second entry
        model_nir.backbone.conv1 = nn.Conv2d(
                1, model_rgb.backbone.conv1.out_channels, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False
        )

        # Adding the weight of the green channel of the pretrained weight (if load)
        # to the nir conv1 (if the image in input is a RGB).
        conv1_w = model_rgb.backbone._modules['conv1'].weight.detach()#.numpy()
        #depth = np.random.uniform(low=-1, high=1, size=(64, 1, 7, 7))
        #conv1_w = np.append(conv1_w, depth, axis=1)
        #conv1 = torch.from_numpy(conv1).float()
        model_nir.backbone._modules['conv1'].weight = nn.Parameter(conv1_w[1], requires_grad=True)
        
        if conc_point in ['conv1', 'maxpool']:
            out_channels  = model_rgb.backbone.conv1.out_channels
        elif conc_point == 'layer2':
            out_channels  = model_rgb.backbone.layer2[-1].conv3.out_channels
        elif conc_point == 'layer3':
            out_channels  = model_rgb.backbone.layer3[-1].conv3.out_channels
        elif conc_point == 'layer4':
            out_channels  = model_rgb.backbone.layer4[-1].conv3.out_channels
        else:
            raise ValueError('The layer you want is not in the layers available!')

        self.modelNIR = nn.Sequential(
            *list(model_nir.backbone.children())[:nir_layers[conc_point]]
        )
        self.modelRGB = nn.Sequential(
                *list(model_rgb.backbone.children())[:nir_layers[conc_point]]
        )
        self.leftover = nn.Sequential(
                *list(model_rgb.backbone.children())[nir_layers[conc_point]:]
        )
        self.classifier = model_rgb.classifier

        ####################################################

        # Conv Layer to fit the size of the next layer
        self.conv1x1 = nn.Conv2d(
                in_channels = out_channels*2, out_channels = out_channels, kernel_size=1
        )
        
        del model_nir, model_rgb

    def forward(self, inputs):
        x1, x2 = inputs
        input_shape = x1.shape[-2:]
        # Contract: features is a dict of tensors
        result = OrderedDict()

        rgb = self.modelRGB(x1)
        nir = self.modelNIR(x2)
        # Concatenation
        x = torch.cat((rgb, nir), dim=1)
        # Conv 1x1 need to match the enter of the next layer
        x = self.conv1x1(x)
        # Give the result to the rest of the network
        x = self.leftover(x)
        # Give the result to the classifier
        x = self.classifier(x)
        # See https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py
        # for more info 
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result['out'] = x

        return result

