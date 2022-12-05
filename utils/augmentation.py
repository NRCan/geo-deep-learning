import logging

import torchgeo.transforms
from kornia import augmentation as K
from utils.utils import get_key_def

logging.getLogger(__name__)


def compose_transforms(params,
                       dataset,
                       dontcare=None,
                       ):
    """
    Function to compose the transformations to be applied on every batch.
    :param params: (dict) Parameters found in the yaml config file
    :param dataset: (str) One of 'trn', 'val', 'tst'
    :param dontcare: (int) Value that will be ignored during loss calculation. Used here to pad label
                         if rotation or crop augmentation
    :return: (obj) PyTorch's compose object of the transformations to be applied.
    """
    lst_trans = []
    keys = ["image", "mask"] if dataset != 'inference' else ["image"]
    norm_mean = get_key_def('mean', params['augmentation']['normalization'])
    norm_std = get_key_def('std', params['augmentation']['normalization'])

    if norm_mean and norm_std:
        if max(norm_mean) > 1 or max(norm_std) > 1:
            logging.error(f"Means and stds should be calculated over raster data scaled between 0 and 1."
                          f"Provided values (means: {norm_mean}, stds: {norm_std}) indicate these may be "
                          f"calculated over original range (ex.: 0-255)")
        lst_trans.append(K.Normalize(mean=list(params['augmentation']['normalization']['mean']),
                                     std=list(params['augmentation']['normalization']['std']),
                                     keepdim=True))

    if dataset == 'trn':
        noise = get_key_def('noise', params['augmentation'], None)
        if noise:
            lst_trans.append(K.RandomGaussianNoise(std=noise, keepdim=True))
        hflip = get_key_def('hflip_prob', params['augmentation'], None)
        rotate_prob = get_key_def('rotate_prob', params['augmentation'], None)
        rotate_limit = get_key_def('rotate_limit', params['augmentation'], None)
        crop_size = get_key_def('crop_size', params['augmentation'], None)
        if hflip:
            lst_trans.append(K.RandomHorizontalFlip(p=hflip, keepdim=True))
        if rotate_limit and rotate_prob:
            lst_trans.append(K.RandomRotation(degrees=rotate_limit, p=rotate_prob, keepdim=True))
        if crop_size:
            lst_trans.append(K.RandomCrop(size=(crop_size, crop_size), fill=dontcare, keepdim=True))

    if lst_trans:
        return torchgeo.transforms.AugmentationSequential(*lst_trans, data_keys=keys)
    else:
        return None
