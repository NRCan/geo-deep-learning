from collections import OrderedDict
from typing import Union, List
import logging

from hydra.utils import instantiate
import torch
import torch.nn as nn
from omegaconf import DictConfig

from utils.utils import get_key_def

logging.getLogger(__name__)


def update_gdl_checkpoint(checkpoint_params):
    bands = ['R', 'G', 'B', 'N']
    old2new = {
        'unet_pretrained': {
            '_target_': 'segmentation_models_pytorch.Unet', 'encoder_name': 'resnext50_32x4d',
            'encoder_depth': 4, 'encoder_weights': 'imagenet', 'decoder_channels': [256, 128, 64, 32]
        },
        'unet': {
            '_target_': 'models.unet.UNet', 'dropout': False, 'prob': False
        },
        'unet_small': {
            '_target_': 'models.unet.UNetSmall', 'dropout': False, 'prob': False
        },
        'deeplabv3_pretrained': {
            '_target_': 'segmentation_models_pytorch.DeepLabV3', 'encoder_name': 'resnet101',
            'encoder_weights': 'imagenet'
        },
        'deeplabv3_resnet101_dualhead': {
            '_target_': 'models.deeplabv3_dualhead.DeepLabV3_dualhead', 'conc_point': 'conv1',
            'encoder_weights': 'imagenet'
        },
        'deeplabv3+_pretrained': {
            '_target_': 'segmentation_models_pytorch.DeepLabV3Plus', 'encoder_name': 'resnext50_32x4d',
            'encoder_weights': 'imagenet'
        },
    }
    try:
        get_key_def('classes_dict', checkpoint_params['dataset'], expected_type=DictConfig)
        get_key_def('modalities', checkpoint_params['dataset'], expected_type=str)
        get_key_def('model', checkpoint_params, expected_type=DictConfig)
        return checkpoint_params
    except KeyError:
        # covers GDL pre-hydra (<=2.0.0)
        num_classes_ckpt = get_key_def('num_classes', checkpoint_params['global'], expected_type=int)
        num_bands_ckpt = get_key_def('number_of_bands', checkpoint_params['global'], expected_type=int)
        model_name = get_key_def('model_name', checkpoint_params['global'], expected_type=str)
        try:
            model_ckpt = old2new[model_name]
        except KeyError as e:
            logging.critical(f"\nCouldn't locate yaml configuration for model architecture {model_name} as found "
                             f"in provided checkpoint. Name of yaml may have changed."
                             f"\nError {type(e)}: {e}")
            raise e
        bands_ckpt = ''
        bands_ckpt = bands_ckpt.join([bands[i] for i in range(num_bands_ckpt)])
        checkpoint_params.update({
            'dataset': {
                'modalities': bands_ckpt,
                "classes_dict": {f"class{i + 1}": i + 1 for i in range(num_classes_ckpt)}
            }
        })
        checkpoint_params.update({'model': model_ckpt})
        return checkpoint_params


def define_model_architecture(
        net_params: dict,
        in_channels: int,
        out_classes: int):
    """
    Define the model architecture from config parameters
    @param net_params: (dict) parameters as expected by hydra's instantiate() function
    @param in_channels: number of input channels, i.e. raster bands
    @param out_classes: number of output classes
    @return: torch model as nn.Module object
    """
    return instantiate(net_params, in_channels=in_channels, classes=out_classes)


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
        net_params: dict,
        in_channels: int,
        out_classes: int,
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
        net_params=net_params,
        in_channels=in_channels,
        out_classes=out_classes,
    )
    model = to_dp_model(model=model, devices=devices[1:]) if len(devices) > 1 else model
    model.to(main_device)
    if state_dict_path:
        checkpoint = read_checkpoint(state_dict_path)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'], strict=state_dict_strict_load)
    return model
