import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import random
import numpy as np
from skimage import transform
from torchvision import transforms


def compose_transforms(params, dataset):
    """
    Function to compose the transformations to be applied on every batches.
    :param params: (dict) Parameters found in the yaml config file
    :param dataset (str) One of 'trn', 'val', 'tst'
    :return: (obj) PyTorch's compose object of the transformations to be applied.
    """
    lst_trans = []
    if dataset == 'trn':
        if params['training']['augmentation']['hflip_prob']:
            lst_trans.append(HorizontalFlip(prob=params['training']['augmentation']['hflip_prob']))

        if params['training']['augmentation']['rotate_limit'] and params['training']['augmentation']['rotate_prob']:
            lst_trans.append(RandomRotationTarget(limit=params['training']['augmentation']['rotate_limit'],
                                                  prob=params['training']['augmentation']['rotate_prob']))

    lst_trans.append(ToTensorTarget())
    return transforms.Compose(lst_trans)


class RandomRotationTarget(object):
    """Rotate the image and target randomly."""
    def __init__(self, limit, prob):
        self.limit = limit
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            angle = np.random.uniform(-self.limit, self.limit)
            sat_img = transform.rotate(sample['sat_img'], angle, preserve_range=True)
            map_img = transform.rotate(sample['map_img'], angle, preserve_range=True)
            return {'sat_img': sat_img, 'map_img': map_img}
        else:
            return sample


class HorizontalFlip(object):
    """Flip the input image and reference map horizontally, with a probability."""
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sat_img = sample['sat_img'][:, ::-1]
            map_img = sample['map_img'][:, ::-1]
            return {'sat_img': sat_img, 'map_img': map_img}
        else:
            return sample


class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img = np.float32(np.transpose(sample['sat_img'], (2, 0, 1)))
        map_img = np.int64(sample['map_img'])
        return {'sat_img': torch.from_numpy(sat_img), 'map_img': torch.from_numpy(map_img)}
