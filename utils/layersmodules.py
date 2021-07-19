import logging

import torch
import copy
from torch import nn
import torchvision.models as models
from collections import OrderedDict
from torch.nn import functional as F

logging.getLogger(__name__)

nir_layers = {'conv1': 1, 'maxpool': 4, 'layer2': 6, 'layer3': 7, 'layer4': 8}

class LayersEnsemble(nn.Module):
    """
    Class create a model where 2 heads concatenate at a specific point.

    This class copy the *model* in input when initialize, copy it,
    change the input of the second *model* for the input dimension
    of the second entry. Concatenate the 2 *models* at a specific
    point chosen when initialize. Like that we have a new *model*
    that take 2 entries with differents depth and combine it to
    have an ouput with the depth of corresponding at the number of
    classes.

    :param model: Model chose to be duplicate and concatenated.
    :param conc_point: Position where the two models are concatenated (see `.yaml` for available point by model).
    :type conc_point: str

    .. note:: Only available for **DeeplabV3** with a backbone of a **Resnet101**.
    .. todo:: Make it more general to be able to be apply on a **UNet** or others.
    """
    def __init__(self, model, conc_point='conv1'):
        """
        In the constructor we instantiate all the part needed for the* model*:

         - ``modelRGB``, containing all the backbone layers **before** the concatenation point.
         - ``modelNIR``, containing all the backbone layers **before** the concatenation point
           and have the number of channels in input change for 1.
         - ``leftover``, containing all the backbone layers **after** the concatenation point.
           For some points, this variables will be empty, since we concatenate after the backbone.
         - ``conv1x1``, an conv2D layer that will be use after the concatenation to go back at
           the depth of one model, since the concatenation operation will double the depth.

        And assign them as member variables.
        """
        super(LayersEnsemble, self).__init__()
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
        # Otherwise it will only copy random initiation weight from the rgb model.
        conv1_w = model_rgb.backbone.conv1.weight.detach() # shape: [64, 3, 7, 7]
        green_weight = conv1_w[:, 1] # [R, G, B] -> [0, 1, 2], shape: [64, 7, 7]
        green_weight.unsqueeze_(1) # shape: [64, 1, 7, 7], otherwise didn't work
        model_nir.backbone.conv1.weight = nn.Parameter(green_weight, requires_grad=True)

        # Concatenation point output channels dimension
        if conc_point in ['conv1', 'maxpool']:
            out_channels = model_rgb.backbone.conv1.out_channels
        elif conc_point == 'layer2':
            out_channels = model_rgb.backbone.layer2[-1].conv3.out_channels
        elif conc_point == 'layer3':
            out_channels = model_rgb.backbone.layer3[-1].conv3.out_channels
        elif conc_point == 'layer4':
            out_channels = model_rgb.backbone.layer4[-1].conv3.out_channels
        else:
            raise ValueError('The layer you want is not in the layers available!')

        # Init all part of the model 
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

        # Conv Layer to fit the size of the next layer
        self.conv1x1 = nn.Conv2d(
                in_channels=out_channels*2, out_channels=out_channels, kernel_size=1
        )

        del model_nir, model_rgb, conv1_w, green_weight

    def forward(self, inputs):
        """
        In the forward function we accept a list of Tensors in input data and return
        a Tensor of output data.

        So in the input list, they should have two tensor, the firt one need to be the
        one containing the **RGB** images, the second one will be the other *modalitie*.

        .. note:: for now we only accept **NIR** as second entry, with a shape of [1, h, w].

        The result of each partial model (RGB and the other) are concatenate on the depth
        dimension and pass by a convolution operation to recover the depth taht match the
        ``leftover`` entry. Follow by the classifier and the interpolation to return a
        Tensor with the same [h, w] then the input Tensor.

        :param inputs: (list) List containing two Tensors
        :return: (tensor) Result
        """
        x1, x2 = self.split_RGB_NIR(inputs)
        # Get the input shape for the interpolation
        input_shape = x1.shape[-2:]
        # Features is a dict of tensors
        result = OrderedDict()
        # Feed the models
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
        # for more info on the use of the interpolation
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result['out'] = x
        return result

    @staticmethod
    def split_RGB_NIR(inputs):
        """
        Split RGB and NIR in input imagery being fed to models for training
        @param inputs: tensors with shape [batch_size x channel x h x w]
        @return: two tensors, one for all but last channel with shape [batch_size x (channel-1) x h x w]
                 (ex.: RGB if RGBN imagery) and the other for NIR with shape [batch_size x 1 x h x w]
        """
        if inputs.shape[1] != 4:
            logging.error(f'Expected 4 band imagery. Got input with {inputs.shape[1]} bands')
        logging.warning(f'Will split inputs in 2 : (1) RGB bands, (2) NIR band')
        inputs_NIR = inputs[:, -1, ...]  # Need to be change for a more elegant way
        inputs_NIR.unsqueeze_(1)  # add a channel to get [:, 1, :, :]
        inputs = inputs[:, :-1, ...]  # Need to be change
        inputs = [inputs, inputs_NIR]
        return inputs
