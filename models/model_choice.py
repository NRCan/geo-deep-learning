from pathlib import Path
import numpy as np
import warnings
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
from models import TernausNet, unet, checkpointed_unet, inception, coordconv
from utils.utils import load_from_checkpoint, get_device_ids, get_key_def

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

def load_checkpoint(filename):
    ''' Loads checkpoint from provided path
    :param filename: path to checkpoint as .pth.tar or .pth
    :return: (dict) checkpoint ready to be loaded into model instance
    '''
    try:
        print(f"=> loading model '{filename}'\n")
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
        warnings.warn("got two class weights for single class defined in configuration file; will assume index 0 = background")
    elif num_classes != len(weights):
        raise ValueError('The number of class weights in the configuration file is different than the number of classes')


def set_hyperparameters(params, num_classes, model, checkpoint, dontcare_val):
    """
    Function to set hyperparameters based on values provided in yaml config file.
    If none provided, default functions values may be used.
    :param params: (dict) Parameters found in the yaml config file
    :param num_classes: (int) number of classes for current task
    :param model: initialized model
    :param checkpoint: (dict) state dict as loaded by model_choice.py
    :return: model, criterion, optimizer, lr_scheduler, num_gpus
    """
    # set mandatory hyperparameters values with those in config file if they exist
    lr = get_key_def('learning_rate', params['training'], None, "missing mandatory learning rate parameter")
    weight_decay = get_key_def('weight_decay', params['training'], None, "missing mandatory weight decay parameter")
    step_size = get_key_def('step_size', params['training'], None, "missing mandatory step size parameter")
    gamma = get_key_def('gamma', params['training'], None, "missing mandatory gamma parameter")

    # optional hyperparameters. Set to None if not in config file
    class_weights = torch.tensor(params['training']['class_weights']) if params['training']['class_weights'] else None
    if params['training']['class_weights']:
        verify_weights(num_classes, class_weights)

    # Loss function
    criterion = MultiClassCriterion(loss_type=params['training']['loss_fn'],
                                    ignore_index=dontcare_val,
                                    weight=class_weights)

    # Optimizer
    opt_fn = params['training']['optimizer']
    optimizer = create_optimizer(params=model.parameters(), mode=opt_fn, base_lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    if checkpoint:
        tqdm.write(f'Loading checkpoint...')
        model, optimizer = load_from_checkpoint(checkpoint, model, optimizer=optimizer)

    return model, criterion, optimizer, lr_scheduler


def net(net_params, num_channels, inference=False):
    """Define the neural net"""
    model_name = net_params['global']['model_name'].lower()
    num_bands = int(net_params['global']['number_of_bands'])
    msg = f'Number of bands specified incompatible with this model. Requires 3 band data.'
    train_state_dict_path = get_key_def('state_dict_path', net_params['training'], None)
    pretrained = get_key_def('pretrained', net_params['training'], True) if not inference else False
    dropout = get_key_def('dropout', net_params['training'], False)
    dropout_prob = get_key_def('dropout_prob', net_params['training'], 0.5)
    dontcare_val = get_key_def("ignore_index", net_params["training"], -1)
    num_devices = net_params['global']['num_gpus']

    if dontcare_val == 0:
        warnings.warn("The 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero;"
                      " all valid class indices should be consecutive, and start at 0. The 'dontcare' value"
                      " will be remapped to -1 while loading the dataset, and inside the config from now on.")
        net_params["training"]["ignore_index"] = -1

    # TODO: find a way to maybe implement it in classification one day
    if 'concatenate_depth' in net_params['global']:
        # Read the concatenation point
        conc_point = net_params['global']['concatenate_depth']

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
        assert (num_bands == 3 or num_bands == 4), msg
        if num_bands == 3:
            print('Finetuning pretrained deeplabv3 with 3 bands')
            model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=True)
            classifier = list(model.classifier.children())
            model.classifier = nn.Sequential(*classifier[:-1])
            model.classifier.add_module('4', nn.Conv2d(classifier[-1].in_channels, num_channels, kernel_size=(1, 1)))
        elif num_bands == 4:
            print('Finetuning pretrained deeplabv3 with 4 bands')
            print('Testing with 4 bands, concatenating at {}.'.format(conc_point))

            model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=True)

            if conc_point=='baseline':
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
                #conv1 = model.backbone._modules['conv1'].weight.detach().numpy()
                #depth = np.random.uniform(low=-1, high=1, size=(64, 1, 7, 7))
                #conv1 = np.append(conv1, depth, axis=1)
                #conv1 = torch.from_numpy(conv1).float()
                #model.backbone._modules['conv1'].weight = nn.Parameter(conv1, requires_grad=True)
                ###################
                model = LayersEnsemble(model, conc_point=conc_point)

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

        return model, checkpoint, model_name

    else:

        if train_state_dict_path is not None:
            assert Path(train_state_dict_path).is_file(), f'Could not locate checkpoint at {train_state_dict_path}'
            checkpoint = load_checkpoint(train_state_dict_path)
        else:
            checkpoint = None
        assert num_devices is not None and num_devices >= 0, "missing mandatory num gpus parameter"
        # list of GPU devices that are available and unused. If no GPUs, returns empty list
        lst_device_ids = get_device_ids(num_devices) if torch.cuda.is_available() else []
        num_devices = len(lst_device_ids) if lst_device_ids else 0
        device = torch.device(f'cuda:{lst_device_ids[0]}' if torch.cuda.is_available() and lst_device_ids else 'cpu')
        print(f"Number of cuda devices requested: {net_params['global']['num_gpus']}. Cuda devices available: {lst_device_ids}\n")
        if num_devices == 1:
            print(f"Using Cuda device {lst_device_ids[0]}\n")
        elif num_devices > 1:
            print(f"Using data parallel on devices: {str(lst_device_ids)[1:-1]}. Main device: {lst_device_ids[0]}\n") # TODO: why are we showing indices [1:-1] for lst_device_ids?
            try:  # For HPC when device 0 not available. Error: Invalid device id (in torch/cuda/__init__.py).
                model = nn.DataParallel(model,
                                        device_ids=lst_device_ids)  # DataParallel adds prefix 'module.' to state_dict keys
            except AssertionError:
                warnings.warn(f"Unable to use devices {lst_device_ids}. Trying devices {list(range(len(lst_device_ids)))}")
                device = torch.device('cuda:0')
                lst_device_ids = range(len(lst_device_ids))
                model = nn.DataParallel(model,
                                        device_ids=lst_device_ids)  # DataParallel adds prefix 'module.' to state_dict keys
        else:
            warnings.warn(f"No Cuda device available. This process will only run on CPU\n")
        tqdm.write(f'Setting model, criterion, optimizer and learning rate scheduler...\n')
        try:  # For HPC when device 0 not available. Error: Cuda invalid device ordinal.
            model.to(device)
        except RuntimeError:
            warnings.warn(f"Unable to use device. Trying device 0...\n")
            device = torch.device(f'cuda:0' if torch.cuda.is_available() and lst_device_ids else 'cpu')
            model.to(device)

        model, criterion, optimizer, lr_scheduler = set_hyperparameters(net_params, num_channels, model, checkpoint, dontcare_val)
        criterion = criterion.to(device)

        return model, model_name, criterion, optimizer, lr_scheduler
