import os
from pathlib import Path

import torch
import warnings
import torchvision.models as models
from models import TernausNet, unet, checkpointed_unet, inception, coordconv
from utils.utils import chop_layer, get_key_def


def load_checkpoint(filename):
    ''' Loads checkpoint from provided path
    :param filename: path to checkpoint as .pth.tar or .pth
    :return: (dict) checkpoint ready to be loaded into model instance
    '''
    try:
        print(f"=> loading model '{filename}'\n")
        checkpoint = torch.load(filename) if torch.cuda.is_available() else torch.load(filename, map_location='cpu')
        # For loading external models with different structure in state dict. May cause problems when trying to load optimizer
        if 'model' not in checkpoint.keys():
            temp_checkpoint = {}
            temp_checkpoint['model'] = {k: v for k, v in checkpoint.items()}    # Place entire state_dict inside 'model' key
            del checkpoint
            checkpoint = temp_checkpoint
        return checkpoint
    except FileNotFoundError:
        raise FileNotFoundError(f"=> No model found at '{filename}'")


def net(net_params, num_channels, inference=False):
    """Define the neural net"""
    model_name = net_params['global']['model_name'].lower()
    num_bands = int(net_params['global']['number_of_bands'])
    msg = f'Number of bands specified incompatible with this model. Requires 3 band data.'
    train_state_dict_path = get_key_def('state_dict_path', net_params['training'], None)
    pretrained = get_key_def('pretrained', net_params['training'], True) if not inference else False
    dropout = get_key_def('dropout', net_params['training'], False)
    dropout_prob = get_key_def('dropout_prob', net_params['training'], 0.5)

    if model_name == 'unetsmall':
        model = unet.UNetSmall(num_channels, num_bands, dropout, dropout_prob)
    elif model_name == 'unet':
        model = unet.UNet(num_channels, num_bands, dropout, dropout_prob)
    elif model_name == 'ternausnet':
        assert num_bands == 3, msg
        model = TernausNet.ternausnet(num_channels)
    elif model_name == 'checkpointed_unet':
        model = checkpointed_unet.UNetSmall(num_channels, num_bands, dropout, dropout_prob)
    elif model_name == 'inception':
        model = inception.Inception3(num_channels, num_bands)
    elif model_name == 'fcn_resnet101':
        assert num_bands == 3, msg
        model = models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=num_channels,
                                                  aux_loss=None)
    elif model_name == 'deeplabv3_resnet101':
        try:
            model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, in_channels=num_bands,
                                                            num_classes=num_channels, aux_loss=None)
        except:
            assert num_bands==3, 'Edit torchvision scripts segmentation.py and resnet.py to build deeplabv3_resnet ' \
                                 'with more or less than 3 bands'
            model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True,
                                                            num_classes=num_channels, aux_loss=None)
    else:
        raise ValueError(f'The model name {model_name} in the config.yaml is not defined.')

    coordconv_convert = get_key_def('coordconv_convert', net_params['global'], False)
    if coordconv_convert:
        centered = get_key_def('coordconv_centered', net_params['global'], True)
        normalized = get_key_def('coordconv_normalized', net_params['global'], True)
        noise = get_key_def('coordconv_noise', net_params['global'], None)
        radius_channel = get_key_def('coordconv_radius_channel', net_params['global'], False)
        scale = get_key_def('coordconv_scale', net_params['global'], 1.0)
        # note: this operation will not attempt to preserve already-loaded model parameters!
        model = coordconv.swap_coordconv_layers(model, centered=centered, normalized=normalized, noise=noise,
                                                radius_channel=radius_channel, scale=scale)

    if inference:
        state_dict_path = net_params['inference']['state_dict_path']
        assert Path(net_params['inference']['state_dict_path']).is_file(), f"Could not locate {net_params['inference']['state_dict_path']}"
        checkpoint = load_checkpoint(state_dict_path)
    elif train_state_dict_path is not None:
        assert Path(train_state_dict_path).is_file()
        checkpoint = load_checkpoint(train_state_dict_path)
    elif pretrained and (model_name == ('deeplabv3_resnet101' or 'fcn_resnet101')):
        print(f'Retrieving coco checkpoint for {model_name}...\n')
        if model_name == 'deeplabv3_resnet101':  # default to pretrained on coco (21 classes)
            coco_model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
        else:
            coco_model = models.segmentation.fcn_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
        checkpoint = coco_model.state_dict()
        # Place entire state_dict inside 'model' key for compatibility with the rest of GDL workflow
        temp_checkpoint = {}
        temp_checkpoint['model'] = {k: v for k, v in checkpoint.items()}
        del coco_model, checkpoint
        checkpoint = temp_checkpoint
    elif pretrained:
        warnings.warn(f'No pretrained checkpoint found for {model_name}.')
        checkpoint = None
    else:
        checkpoint = None

    return model, checkpoint, model_name
