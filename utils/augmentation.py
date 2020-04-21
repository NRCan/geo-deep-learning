# WARNING: data being augmented may be scaled to (0,1) rather, for example, (0,255). Therefore, implementing radiometric
# augmentations (ex.: changing hue, saturation, brightness, contrast) may give undesired results.
# Scaling process is done in images_to_samples.py l.215

import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import random
import numpy as np
from skimage import transform, exposure
from torchvision import transforms

from utils.utils import get_key_def


def compose_transforms(params, dataset, type=''):
    """
    Function to compose the transformations to be applied on every batches.
    :param params: (dict) Parameters found in the yaml config file
    :param dataset: (str) One of 'trn', 'val', 'tst'
    :param type: (str) One of 'geometric', 'radiometric'
    :return: (obj) PyTorch's compose object of the transformations to be applied.
    """
    lst_trans = []
    if dataset == 'trn':

        if type == 'radiometric':
            radiom_trim_range = get_key_def('radiom_trim_range', params['training']['augmentation'], None)
            brightness_contrast_range = get_key_def('brightness_contrast_range', params['training']['augmentation'], None)
            noise = get_key_def('noise', params['training']['augmentation'], None)

            if radiom_trim_range:  # Contrast stretching
                lst_trans.append(RadiometricTrim(range=radiom_trim_range))  # FIXME: test this. Assure compatibility with CRIM devs (don't trim metadata)

            if brightness_contrast_range:
                # lst_trans.append()
                pass

            if noise:
                # lst_trans.append()
                pass

        elif type == 'geometric':
            geom_scale_range = get_key_def('geom_scale_range', params['training']['augmentation'], None)
            hflip = get_key_def('hflip_prob', params['training']['augmentation'], None)
            rotate_prob = get_key_def('rotate_prob', params['training']['augmentation'], None)
            rotate_limit = get_key_def('rotate_limit', params['training']['augmentation'], None)
            crop_size = get_key_def('crop_size', params['training']['augmentation'], None)

            if geom_scale_range:
                lst_trans.append(GeometricScale(range=geom_scale_range))  # FIXME: test this

            if hflip:
                lst_trans.append(HorizontalFlip(prob=params['training']['augmentation']['hflip_prob']))

            if rotate_limit and rotate_prob:
                lst_trans.append(RandomRotationTarget(limit=rotate_limit, prob=rotate_prob))

            if crop_size:
                lst_trans.append(RandomCrop(sample_size=crop_size))  # FIXME: test this

            lst_trans.append(ToTensorTarget())  # Send channels first, convert numpy array to torch tensor

    else:
        lst_trans.append(ToTensorTarget())  # Send channels first, convert numpy array to torch tensor

    return transforms.Compose(lst_trans)


class RadiometricTrim(object):
    """Randomly trim values left and right in a certain range (%)."""
    def __init__(self, range):
        self.range = range

    def __call__(self, sample):
        trim = round(random.uniform(range[0], range[-1]), 1)
        perc_left, perc_right = np.percentile(trim, 100-trim)
        sat_img = exposure.rescale_intensity(sample['sat_img'], in_range=(perc_left, perc_right))
        return {'sat_img': sat_img, 'map_img': sample['map_img']}


class GeometricScale(object):
    """Randomly resize image according to a certain range."""
    def __init__(self, range):
        self.range = range

    def __call__(self, sample):
        scale_factor = round(random.uniform(range[0], range[-1]), 1)
        output_width = sample['sat_img'].shape[0] * scale_factor
        output_height =  sample['sat_img'].shape[1] * scale_factor
        sat_img = transform.resize(sample['sat_img'], output_shape=(output_height, output_width))
        map_img = transform.resize(sample['map_img'], output_shape=(output_height, output_width))
        return {'sat_img': sat_img, 'map_img': map_img}


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
            sat_img = np.ascontiguousarray(sample['sat_img'][:, ::-1, ...])
            map_img = np.ascontiguousarray(sample['map_img'][:, ::-1, ...])
            return {'sat_img': sat_img, 'map_img': map_img}
        else:
            return sample


class RandomCrop(object):  # FIXME: delete overlap in samples_prep (images_to_samples, l.106)
    """Randomly crop image according to a certain dimension."""
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def __call__(self, sample):
        ########################################################################
        # select a (sample_size x sample_size) random crop from the img and label:
        ########################################################################
        start_x = np.random.randint(low=0, high=(sample['sat_img'].shape[0] - self.sample_size))
        end_x = start_x + self.sample_size
        start_y = np.random.randint(low=0, high=(sample['sat_img'].shape[1] - self.sample_size))
        end_y = start_y + self.sample_size

        sat_img = sample['sat_img'][start_y:end_y, start_x:end_x]  # ex.: (shape: (256, 256, 3))
        map_img = sample['map_img'][start_y:end_y, start_x:end_x]  # ex.: (shape: (256, 256))
        return {'sat_img': sat_img, 'map_img': map_img}


class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # FIXME: add default trim (e.g. 0.1% or 0.5%)
        #  FIXME: scale between 0 and 1 HERE, not in samples_prep.
        sat_img = np.float32(np.transpose(sample['sat_img'], (2, 0, 1)))
        map_img = np.int64(sample['map_img'])
        return {'sat_img': torch.from_numpy(sat_img), 'map_img': torch.from_numpy(map_img)}
