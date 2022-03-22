from collections import OrderedDict
from typing import Union, List
import logging

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from models.deeplabv3_dualhead import DeepLabV3_dualhead
from models import unet, checkpointed_unet

logging.getLogger(__name__)

lm_smp = {
    'pan_pretrained': {
        'fct': smp.PAN, 'params': {
            'encoder_name': 'se_resnext101_32x4d',
        }},
    'unet_pretrained': {
        'fct': smp.Unet, 'params': {
            'encoder_name': 'resnext50_32x4d',
            'encoder_depth': 5,
        }},
    'unet_pretrained_101': {
        'fct': smp.Unet, 'params': {
            'encoder_name': 'resnext101_32x8d',
            'encoder_depth': 4,
            'decoder_channels': [256, 128, 64, 32]
        }},
    'unet_plus_pretrained': {
        'fct': smp.UnetPlusPlus, 'params': {
            'encoder_name': 'se_resnext50_32x4d',
            'encoder_depth': 4,
            'decoder_channels': [256, 128, 64, 32],
            'decoder_attention_type': 'scse'
        }},
    'deeplabv3+_pretrained': {
        'fct': smp.DeepLabV3Plus, 'params': {
            'encoder_name': 'resnext50_32x4d',
        }},
    'spacenet_unet_efficientnetb5_pretrained': {
        'fct': smp.Unet, 'params': {
            'encoder_name': "efficientnet-b5",
        }},
    'spacenet_unet_senet152_pretrained': {
        'fct': smp.Unet, 'params': {
            'encoder_name': 'senet154',
        }},
    'spacenet_unet_baseline_pretrained': {
        # In the article of SpaceNet, the baseline is originaly pretrained on 'SN6 PS-RGB Imagery'.
        'fct': smp.Unet, 'params': {
            'encoder_name': 'vgg11',
        }},
}
try:
    lm_smp['manet_pretrained'] = {
        # https://ieeexplore.ieee.org/abstract/document/9201310
        'fct': smp.MAnet, 'params': {
            'encoder_name': 'resnext50_32x4d'}}
except AttributeError:
    logging.exception("Couldn't load MAnet from segmentation models pytorch package. Check installed version")


def define_model_architecture(
    model_name: str,
    num_bands: int,
    num_channels: int,
    dropout_prob: float = False,
    conc_point: str = None,
):
    """
    Define the model architecture from config parameters
    """
    dropout = True if dropout_prob else False
    if model_name == 'unetsmall':
        model = unet.UNetSmall(num_channels, num_bands, dropout, dropout_prob)
    elif model_name == 'unet':
        model = unet.UNet(num_channels, num_bands, dropout, dropout_prob)
    elif model_name == 'checkpointed_unet':
        model = checkpointed_unet.UNetSmall(num_channels, num_bands, dropout, dropout_prob)
    elif model_name == 'deeplabv3_pretrained':
        model = smp.DeepLabV3(encoder_name='resnet101', in_channels=num_bands, classes=num_channels)
    elif model_name == 'deeplabv3_resnet101_dualhead':
        model = DeepLabV3_dualhead(encoder_name='resnet101', in_channels=num_bands, classes=num_channels,
                                   conc_point=conc_point)
    elif model_name in lm_smp.keys():
        lsmp = lm_smp[model_name]
        # TODO: add possibility of our own weights
        lsmp['params']['encoder_weights'] = "imagenet" if 'pretrained' in model_name.split("_") else None
        lsmp['params']['in_channels'] = num_bands
        lsmp['params']['classes'] = num_channels
        lsmp['params']['activation'] = None

        model = lsmp['fct'](**lsmp['params'])
    else:
        raise logging.critical(ValueError(f'\nThe model name {model_name} in the config.yaml is not defined.'))

    return model


def read_checkpoint(filename):
    """
    Loads checkpoint from provided path to GDL's expected format,
    ie model's state dictionary should be under "model_state_dict" and
    optimizer's state dict should be under "optimizer_state_dict" as suggested by pytorch:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
    :param filename: path to checkpoint as .pth.tar or .pth
    :return: (dict) checkpoint ready to be loaded into model instance
    """
    if not filename:
        logging.warning(f"No path to checkpoint provided.")
        return None
    try:
        logging.info(f"\n=> loading model '{filename}'")
        # For loading external models with different structure in state dict.
        checkpoint = torch.load(filename, map_location='cpu')
        if 'model_state_dict' not in checkpoint.keys():
            val_set = set()
            for val in checkpoint.values():
                val_set.add(type(val))
            if len(val_set) == 1 and list(val_set)[0] == torch.Tensor:
                # places entire state_dict inside expected key
                new_checkpoint = OrderedDict()
                new_checkpoint['model_state_dict'] = OrderedDict({k: v for k, v in checkpoint.items()})
                del checkpoint
                checkpoint = new_checkpoint
            # Covers gdl's checkpoints at version <=2.0.1
            elif 'model' in checkpoint.keys():
                checkpoint['model_state_dict'] = checkpoint['model']
                del checkpoint['model']
            else:
                raise ValueError(f"GDL cannot find weight in provided checkpoint")
        if 'optimizer_state_dict' not in checkpoint.keys():
            try:
                # Covers gdl's checkpoints at version <=2.0.1
                checkpoint['optimizer_state_dict'] = checkpoint['optimizer']
                del checkpoint['optimizer']
            except KeyError:
                logging.critical(f"No optimizer state dictionary was found in provided checkpoint")
        return checkpoint
    except FileNotFoundError:
        raise logging.critical(FileNotFoundError(f"\n=> No model found at '{filename}'"))


def adapt_checkpoint_to_dp_model(checkpoint: dict, model: Union[nn.Module, nn.DataParallel]):
    """
    Adapts a generic checkpoint to be loaded to a DataParallel model.
    See: https://github.com/bearpaw/pytorch-classification/issues/27
    Also: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3

    Args:
        checkpoint: a dict containing parameters and
            persistent buffers under "model_state_dict" key
        model: a pytorch model to adapt checkpoint to (especially if model is a nn.DataParallel class)
    """
    if isinstance(model, nn.DataParallel):
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v
        new_state_dict['model_state_dict'] = {'module.' + k: v for k, v in checkpoint['model_state_dict'].items()}
        del checkpoint
        checkpoint = {}
        checkpoint['model_state_dict'] = new_state_dict['model_state_dict']
    elif isinstance(model, nn.Module):
        logging.info(f"Provided model is not a DataParallel model. No need to adapt checkpoint to ordinary model")
    else:
        return ValueError(f"Cannot adapt checkpoint to model of class '{type(model)}'."
                          f"\nThis adapter only supports 'nn.Module' and 'nn.DataParallel' models")
    return checkpoint


def to_dp_model(model, devices: List):
    """
    Converts a model to a DataParallel model given a list of device ids as integers
    @param model: nn.Module (pytorch model)
    @param devices: list of devices ids as integers
    @return:
    """
    if not devices or len(devices) == 1:
        return model
    logging.info(f"\nUsing data parallel on devices: {devices}. "
                 f"Main device: 'cuda:{devices[0]}'")
    try:  # For HPC when device 0 not available. Error: Invalid device id (in torch/cuda/__init__.py).
        model = nn.DataParallel(model, device_ids=devices)
    except AssertionError:
        logging.warning(f"\nUnable to use devices with ids {devices}"
                        f"Trying devices with ids {list(range(len(devices)))}")
        model = nn.DataParallel(model, device_ids=list(range(len(devices))))
    return model


def define_model(
        model_name,
        num_bands,
        num_classes,
        dropout_prob: float = 0.5,
        conc_point: str = None,
        main_device: str = 'cpu',
        devices: List = [],
        state_dict_path: str = None,
        state_dict_strict_load: bool = True,
):
    """
    Defines model's architecture with weights from provided checkpoint and pushes to device(s)
    @return:
    """
    model = define_model_architecture(
        model_name=model_name,
        num_bands=num_bands,
        num_channels=num_classes,
        dropout_prob=dropout_prob,
        conc_point=conc_point
    )
    model = to_dp_model(model=model, devices=devices[1:]) if len(devices) > 1 else model
    model.to(main_device)
    if state_dict_path:
        checkpoint = read_checkpoint(state_dict_path)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'], strict=state_dict_strict_load)
    return model
