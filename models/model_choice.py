from collections import OrderedDict
from typing import Union, List
import logging

from hydra.utils import instantiate
import torch.nn as nn

from utils.utils import read_checkpoint

logging.getLogger(__name__)


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
