import logging
from typing import Optional

import torch

from segmentation_models_pytorch import DeepLabV3
from torch import nn
from collections import OrderedDict

logging.getLogger(__name__)


class DeepLabV3_dualhead(nn.Module):
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
    
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_channels: int = 256,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 8,
            aux_params: Optional[dict] = None,
            conc_point: Optional[str] = 'conv1',
    ):
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
        super().__init__()

        self.nir_layers = {'conv1': 1, 'maxpool': 4, 'layer2': 6, 'layer3': 7, 'layer4': 8}

        if not in_channels == 4:
            raise NotImplementedError(f"The dual head Deeplabv3 is implemented only for 4 band imagery. "
                                      f"\nGot {in_channels} bands")

        logging.info(f'\nFinetuning pretrained deeplabv3 with 4 input channels (imagery bands). '
                     f'Concatenation point: "{conc_point}"')

        # Init model
        model_rgb = DeepLabV3(encoder_name, encoder_depth, encoder_weights, decoder_channels, in_channels-1, classes,
                              activation, upsampling, aux_params)
        # SMP carries over pretrained weights to models with input channels != 3
        # https://github.com/qubvel/segmentation_models.pytorch#input-channels
        model_nir = DeepLabV3(encoder_name, encoder_depth, encoder_weights, decoder_channels, 1, classes, activation,
                              upsampling, aux_params)

        # Concatenation point output channels dimension
        if conc_point in ['conv1', 'maxpool']:
            out_channels = model_rgb.encoder.conv1.out_channels
        elif conc_point == 'layer2':
            out_channels = model_rgb.encoder.layer2[-1].conv3.out_channels
        elif conc_point == 'layer3':
            out_channels = model_rgb.encoder.layer3[-1].conv3.out_channels
        elif conc_point == 'layer4':
            out_channels = model_rgb.encoder.layer4[-1].conv3.out_channels
        else:
            raise ValueError('The layer you want is not in the layers available!')

        # Init all parts of the model
        # An alternate implementation could use torchvision.models._utils.IntermediateLayerGetter
        # https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
        self.modelNIR = nn.Sequential(
            *list(model_nir.encoder.children())[:self.nir_layers[conc_point]]
        )
        self.modelRGB = nn.Sequential(
                *list(model_rgb.encoder.children())[:self.nir_layers[conc_point]]
        )

        # Conv Layer to fit the size of the first leftover layer
        self.conv1x1 = nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1
        )

        self.leftover = nn.Sequential(
                *list(model_rgb.encoder.children())[self.nir_layers[conc_point]:]
        )
        self.decoder = model_rgb.decoder

        self.segmentation_head = model_rgb.segmentation_head

        del model_nir, model_rgb

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
        x3 = torch.cat((rgb, nir), dim=1)
        # Conv 1x1 need to match the input dimensions of the next layer
        x4 = self.conv1x1(x3)
        # Give the result to the rest of the network
        x5 = self.leftover(x4)
        # Give the result to the decoder
        decoder_output = self.decoder(x5)

        # segmentation head performs upsampling to input image size
        masks = self.segmentation_head(decoder_output)

        return masks

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
