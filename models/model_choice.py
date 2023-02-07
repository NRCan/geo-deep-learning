from collections import OrderedDict
from typing import Union, List
import logging

from hydra.utils import instantiate, to_absolute_path
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pandas.io.common import is_url
from torch.hub import load_state_dict_from_url

from utils.utils import update_gdl_checkpoint

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


def read_checkpoint(filename, out_dir: str = 'checkpoints', update=False) -> DictConfig:
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
        if is_url(filename):
            checkpoint = load_state_dict_from_url(url=filename, map_location='cpu', model_dir=to_absolute_path(out_dir))
        else:
            checkpoint = torch.load(f=filename, map_location='cpu')
        # For loading external models with different structure in state dict.
        if 'model_state_dict' not in checkpoint.keys() and 'model' not in checkpoint.keys():
            val_set = set()
            for val in checkpoint.values():
                val_set.add(type(val))
            if len(val_set) == 1 and list(val_set)[0] == torch.Tensor:
                # places entire state_dict inside expected key
                new_checkpoint = OrderedDict()
                new_checkpoint['model_state_dict'] = OrderedDict({k: v for k, v in checkpoint.items()})
                del checkpoint
                checkpoint = new_checkpoint
            else:
                raise ValueError(f"GDL cannot find weight in provided checkpoint")
        elif update:
            checkpoint = update_gdl_checkpoint(checkpoint)
        return checkpoint
    except FileNotFoundError as e:
        logging.critical(f"\n=> No model found at '{filename}'")
        raise e


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
        checkpoint_dict: Union[DictConfig, dict] = None,
        checkpoint_dict_strict_load: bool = True,
) -> nn.Module:
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
    if checkpoint_dict:
        model.load_state_dict(state_dict=checkpoint_dict['model_state_dict'], strict=checkpoint_dict_strict_load)
    return model
