import logging

import torch
from hydra.utils import instantiate


def define_loss(loss_params, class_weights):
    """
    Defines a loss criterion from config parameters as expected by hydra's intanstiate utility
    @return:
    """
    class_weights = torch.tensor(class_weights) if class_weights else None
    # Loss function
    if loss_params['_target_'] in ['torch.nn.CrossEntropyLoss', 'losses.focal_loss.FocalLoss',
                               'losses.ohem_loss.OhemCrossEntropy2d']:
        criterion = instantiate(loss_params, weight=class_weights)
    else:
        criterion = instantiate(loss_params)
    return criterion


def verify_weights(num_classes, weights):
    """Verifies that the length of weights for loss function equals the number of classes if any are given
    Args:
        num_classes: number of classes defined in the configuration file
        weights: weights defined in the configuration file
    """
    if num_classes == 1 and len(weights) == 2:
        logging.warning(
            "got two class weights for single class defined in configuration file; will assume index 0 = background")
    elif num_classes != len(weights):
        raise ValueError(f'The number of class weights {len(weights)} '
                         f'in the configuration file is different than the number of classes {num_classes}')