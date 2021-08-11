import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.models as models
###############################
from utils.layersmodules import LayersEnsemble
###############################
from tqdm import tqdm
from utils.optimizer import create_optimizer
from losses import MultiClassCriterion
import torch.optim as optim
from models import TernausNet, unet, checkpointed_unet, inception, coordconv, MECnet
from utils.utils import load_from_checkpoint, get_device_ids, get_key_def, defaults_from_params

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
    'fpn_pretrained': {
        'fct': smp.FPN, 'params': {
            'encoder_name': 'resnext50_32x4d',
        }},
    'pspnet_pretrained': {
        'fct': smp.PSPNet, 'params': {
            'encoder_name': "resnext50_32x4d",
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
# try:
#     lm_smp['manet_pretrained'] = {
#         # https://ieeexplore.ieee.org/abstract/document/9201310
#         'fct': smp.MAnet, 'params': {
#             'encoder_name': 'resnext50_32x4d'}}
# except AttributeError:
#     logging.exception("Couldn't load MAnet from segmentation models pytorch package. Check installed version")


def load_checkpoint(filename):
    ''' Loads checkpoint from provided path
    :param filename: path to checkpoint as .pth.tar or .pth
    :return: (dict) checkpoint ready to be loaded into model instance
    '''
    try:
        logging.info(f"=> loading model '{filename}'\n")
        # For loading external models with different structure in state dict. May cause problems when trying to load optimizer
        checkpoint = torch.load(filename, map_location='cpu')
        if 'model' not in checkpoint.keys():
            temp_checkpoint = {}
            temp_checkpoint['model'] = {k: v for k, v in checkpoint.items()}    # Place entire state_dict inside 'model' key
            del checkpoint
            checkpoint = temp_checkpoint
        return checkpoint
    except FileNotFoundError:
        raise FileNotFoundError(f"=> No model found at '{filename}'")


def verify_weights(num_classes, weights):
    """Verifies that the number of weights equals the number of classes if any are given
    Args:
        num_classes: number of classes defined in the configuration file
        weights: weights defined in the configuration file
    """
    if num_classes == 1 and len(weights) == 2:
        logging.warning("got two class weights for single class defined in configuration file; will assume index 0 = background")
    elif num_classes != len(weights):
        raise ValueError('The number of class weights in the configuration file is different than the number of classes')


def set_hyperparameters(params,
                        num_classes,
                        model,
                        checkpoint,
                        dontcare_val,
                        loss_fn,
                        optimizer,
                        class_weights=None,
                        inference: str = ''):
    """
    Function to set hyperparameters based on values provided in yaml config file.
    If none provided, default functions values may be used.
    :param params: (dict) Parameters found in the yaml config file
    :param num_classes: (int) number of classes for current task
    :param model: initialized model
    :param checkpoint: (dict) state dict as loaded by model_choice.py
    :param dontcare_val: value in label to ignore during loss calculation
    :param loss_fn: loss function
    :param optimizer: optimizer function
    :param class_weights: class weights for loss function
    :param inference: (str) path to inference checkpoint (used in load_from_checkpoint())
    :return: model, criterion, optimizer, lr_scheduler, num_gpus
    """
    # set mandatory hyperparameters values with those in config file if they exist
    lr = get_key_def('learning_rate', params['training'], None)
    weight_decay = get_key_def('weight_decay', params['training'], None)
    step_size = get_key_def('step_size', params['training'], None)
    gamma = get_key_def('gamma', params['training'], None)

    class_weights = torch.tensor(class_weights) if class_weights else None

    # Loss function
    criterion = MultiClassCriterion(loss_type=loss_fn,
                                    ignore_index=dontcare_val,
                                    weight=class_weights)

    # Optimizer
    opt_fn = optimizer
    optimizer = create_optimizer(params=model.parameters(), mode=opt_fn, base_lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    if checkpoint:
        tqdm.write(f'Loading checkpoint...')
        model, optimizer = load_from_checkpoint(checkpoint, model, optimizer=optimizer, inference=inference)

    return model, criterion, optimizer, lr_scheduler


def net(model_name: str,
        num_bands: int,
        num_channels: int,
        dontcare_val: int,
        num_devices: int,
        train_state_dict_path: str = None,
        pretrained: bool = True,
        dropout_prob: float = False,
        loss_fn: str = None,
        optimizer: str = None,
        class_weights: Sequence = None,
        net_params=None,
        conc_point: str = None,
        coordconv_params=None,
        inference_state_dict: str = None):
    """Define the neural net"""
    msg = f'Number of bands specified incompatible with this model. Requires 3 band data.'
    pretrained = False if train_state_dict_path or inference_state_dict else pretrained
    dropout = True if dropout_prob else False

    if model_name == 'mecnet':
        assert num_bands == 3, msg
        model = MECnet.MECNet()# dropout, dropout_prob)

    elif model_name == 'unet':
        model = unet.UNet(num_channels, num_bands, dropout, dropout_prob)
    elif model_name == 'unetsmall':
        model = unet.UNetSmall(num_channels, num_bands, dropout, dropout_prob)
    elif model_name == 'checkpointed_unet':
        model = checkpointed_unet.UNetSmall(num_channels, num_bands, dropout, dropout_prob)

    elif model_name == 'ternausnet':
        if not num_bands == 3:
            raise NotImplementedError(msg)
        model = TernausNet.ternausnet(num_channels)

    elif model_name == 'inception':
        model = inception.Inception3(num_channels, num_bands)

    elif model_name == 'fcn_resnet101':
        if not num_bands == 3:
            raise NotImplementedError(msg)
        model = models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=num_channels, aux_loss=None) # aux_loss ?
    elif model_name == 'deeplabv3_resnet101':
        if not (num_bands == 3 or num_bands == 4):
            raise NotImplementedError(msg)
        if num_bands == 3:
            model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=True)
            classifier = list(model.classifier.children())
            model.classifier = nn.Sequential(*classifier[:-1])
            model.classifier.add_module('4', nn.Conv2d(classifier[-1].in_channels, num_channels, kernel_size=(1, 1)))
        elif num_bands == 4:

            model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=True)

            if conc_point == 'baseline':
                logging.info('Testing with 4 bands, concatenating at {}.'.format(conc_point))
                conv1 = model.backbone._modules['conv1'].weight.detach().numpy()
                depth = np.expand_dims(conv1[:, 1, ...], axis=1)  # reuse green weights for infrared.
                conv1 = np.append(conv1, depth, axis=1)
                conv1 = torch.from_numpy(conv1).float()
                model.backbone._modules['conv1'].weight = nn.Parameter(conv1, requires_grad=True)
                classifier = list(model.classifier.children())
                model.classifier = nn.Sequential(*classifier[:-1])
                model.classifier.add_module(
                    '4', nn.Conv2d(classifier[-1].in_channels, num_channels, kernel_size=(1, 1))
                )
            else:
                classifier = list(model.classifier.children())
                model.classifier = nn.Sequential(*classifier[:-1])
                model.classifier.add_module(
                        '4', nn.Conv2d(classifier[-1].in_channels, num_channels, kernel_size=(1, 1))
                )
                ###################
                # conv1 = model.backbone._modules['conv1'].weight.detach().numpy()
                # depth = np.random.uniform(low=-1, high=1, size=(64, 1, 7, 7))
                # conv1 = np.append(conv1, depth, axis=1)
                # conv1 = torch.from_numpy(conv1).float()
                # model.backbone._modules['conv1'].weight = nn.Parameter(conv1, requires_grad=True)
                ###################
                conc_point = 'conv1' if not conc_point else conc_point
                model = LayersEnsemble(model, conc_point=conc_point)

        logging.info(f'Finetuning pretrained deeplabv3 with {num_bands} input channels (imagery bands). '
                     f'Concatenation point: "{conc_point}"')

    elif model_name in lm_smp.keys():
        lsmp = lm_smp[model_name]
        # TODO: add possibility of our own weights
        lsmp['params']['encoder_weights'] = "imagenet" if 'pretrained' in model_name.split("_") else None
        lsmp['params']['in_channels'] = num_bands
        lsmp['params']['classes'] = num_channels
        lsmp['params']['activation'] = None

        model = lsmp['fct'](**lsmp['params'])

    else:
        raise ValueError(f'The model name {model_name} in the config.yaml is not defined.')


    coordconv_convert = get_key_def('coordconv_convert', coordconv_params, False)
    if coordconv_convert:
        centered = get_key_def('coordconv_centered', coordconv_params, True)
        normalized = get_key_def('coordconv_normalized', coordconv_params, True)
        noise = get_key_def('coordconv_noise', coordconv_params, None)
        radius_channel = get_key_def('coordconv_radius_channel', coordconv_params, False)
        scale = get_key_def('coordconv_scale', coordconv_params, 1.0)
        # note: this operation will not attempt to preserve already-loaded model parameters!
        model = coordconv.swap_coordconv_layers(model, centered=centered, normalized=normalized, noise=noise,
                                                radius_channel=radius_channel, scale=scale)

    if inference_state_dict:
        state_dict_path = inference_state_dict
        checkpoint = load_checkpoint(state_dict_path)

        return model, checkpoint, model_name

    else:
        # checkpoint
        if train_state_dict_path is not None:
            assert Path(train_state_dict_path).is_file(), f'Could not locate checkpoint at {train_state_dict_path}'
            checkpoint = load_checkpoint(train_state_dict_path)
        else:
            checkpoint = None
        # list of GPU devices that are available and unused. If no GPUs, returns empty list
        gpu_devices_dict = get_device_ids(num_devices)
        num_devices = len(gpu_devices_dict.keys())
        device = torch.device(f'cuda:{list(gpu_devices_dict.keys())[0]}' if gpu_devices_dict else 'cpu')
        logging.info(f"Number of cuda devices requested: {num_devices}. "
                     f"Cuda devices available: {list(gpu_devices_dict.keys())}\n")
        if num_devices == 1:
            logging.info(f"Using Cuda device 'cuda:0'")
        elif num_devices > 1:
            logging.info(f"Using data parallel on devices: {list(gpu_devices_dict.keys())[1:]}. "
                         f"Main device: 'cuda:0'")
            try:  # For HPC when device 0 not available. Error: Invalid device id (in torch/cuda/__init__.py).
                # DataParallel adds prefix 'module.' to state_dict keys
                model = nn.DataParallel(model, device_ids=list(gpu_devices_dict.keys()))

                try:  # For HPC when device 0 not available. Error: Cuda invalid device ordinal.
                    model.to(device)
                except AssertionError:
                    logging.exception(f"Unable to use device. Trying device 0...\n")
                    device = torch.device(f'cuda:0' if gpu_devices_dict else 'cpu')
                    model.to(device)

            except AssertionError:
                logging.warning(f"Unable to use devices {gpu_devices_dict}. "
                                f"Trying devices {list(range(len(gpu_devices_dict.keys())))}")
                device = torch.device(f'cuda:0')
                model = nn.DataParallel(model, device_ids=list(range(len(gpu_devices_dict.keys()))))
        else:
            logging.warning(f"No Cuda device available. This process will only run on CPU\n")
        logging.info(f'Setting model, criterion, optimizer and learning rate scheduler...\n')
        try:  # For HPC when device 0 not available. Error: Cuda invalid device ordinal.
            model.to(device)
        except AssertionError:
            logging.exception(f"Unable to use device. Trying device 0...\n")
            device = torch.device(f'cuda:0' if gpu_devices_dict else 'cpu')
            model.to(device)

        model, criterion, optimizer, lr_scheduler = set_hyperparameters(params=net_params,
                                                                        num_classes=num_channels,
                                                                        model=model,
                                                                        checkpoint=checkpoint,
                                                                        dontcare_val=dontcare_val,
                                                                        loss_fn=loss_fn,
                                                                        optimizer=optimizer,
                                                                        class_weights=class_weights,
                                                                        inference=inference_state_dict)
        criterion = criterion.to(device)

        return model, model_name, criterion, optimizer, lr_scheduler
