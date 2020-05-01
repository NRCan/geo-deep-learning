# WARNING: data being augmented may be scaled to (0,1) rather, for example, (0,255). Therefore, implementing radiometric
# augmentations (ex.: changing hue, saturation, brightness, contrast) may give undesired results.
# Scaling process is done in images_to_samples.py l.215
import numbers
import warnings
from typing import Sequence

import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import random
import numpy as np
from scipy import ndimage
from skimage import transform, exposure
from torchvision import transforms

from utils.utils import get_key_def, pad, minmax_scale, pad_diff


def compose_transforms(params, dataset, type='', ignore_index=None):
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
            scale = get_key_def('scale_data', params['global'], None)

            if radiom_trim_range:  # Contrast stretching
                lst_trans.append(RadiometricTrim(range=radiom_trim_range))  # FIXME: test this. Assure compatibility with CRIM devs (don't trim metadata)

            if brightness_contrast_range:
                # lst_trans.append()
                pass

            if noise:
                # lst_trans.append()
                pass

            if scale:
                lst_trans.append(Scale(scale))

        elif type == 'geometric':
            geom_scale_range = get_key_def('geom_scale_range', params['training']['augmentation'], None)
            hflip = get_key_def('hflip_prob', params['training']['augmentation'], None)
            rotate_prob = get_key_def('rotate_prob', params['training']['augmentation'], None)
            rotate_limit = get_key_def('rotate_limit', params['training']['augmentation'], None)
            crop_size = get_key_def('crop_size', params['training']['augmentation'], None)

            if geom_scale_range:  # TODO: test this.
                lst_trans.append(GeometricScale(range=geom_scale_range))

            if hflip:
                lst_trans.append(HorizontalFlip(prob=params['training']['augmentation']['hflip_prob']))

            if rotate_limit and rotate_prob:
                lst_trans.append(RandomRotationTarget(limit=rotate_limit, prob=rotate_prob, ignore_index=ignore_index))

            if crop_size:
                lst_trans.append(RandomCrop(sample_size=crop_size, ignore_index=ignore_index))

            lst_trans.append(ToTensorTarget())  # Send channels first, convert numpy array to torch tensor

    else:
        scale = get_key_def('scale_data', params['global'], None)
        if scale:
            lst_trans.append(Scale(scale))
        lst_trans.append(ToTensorTarget())  # Send channels first, convert numpy array to torch tensor

    return transforms.Compose(lst_trans)


class RadiometricTrim(object):
    """Randomly trim values left and right in a certain range (%)."""
    def __init__(self, range):
        self.range = range

    def __call__(self, sample):
        trim = round(random.uniform(self.range[0], self.range[-1]), 1)
        out_dtype = sample['sat_img_dtype']
        rescaled_sat_img = np.empty(sample['sat_img'].shape, dtype=sample['sat_img'].dtype)
        for band_idx in range(sample['sat_img'].shape[2]):
            band = sample['sat_img'][:, :, band_idx]
            perc_left, perc_right = np.nanpercentile(band, (trim, 100-trim))
            rescaled_band = exposure.rescale_intensity(band, in_range=(perc_left, perc_right), out_range=out_dtype)
            rescaled_sat_img[:, :, band_idx] = rescaled_band
        sample['sat_img'] = rescaled_sat_img
        return sample


class Scale(object):
    """
    Scale array values from range [0,255]  or [0,65535] to values in config (e.g. [0,1])
    Guidelines for pre-processing: http://cs231n.github.io/neural-networks-2/#datapre
    """
    def __init__(self, range):
        if isinstance(range, Sequence) and len(range) == 2:
            self.sc_min = range[0]
            self.sc_max = range[1]
        else:
            raise TypeError('Got inappropriate scale arg')

    @staticmethod
    def range_values_raster(raster, dtype):
        min_val, max_val = np.nanmin(raster), np.nanmax(raster)
        if 'int' in dtype:
            orig_range = (np.iinfo(dtype).min, np.iinfo(dtype).max)
        elif min_val >= 0 and max_val <= 65535:
            orig_range = (0, 65535)
            warnings.warn(f"Values in input image of shape {raster.shape} "
                          f"range from {min_val} to {max_val}."
                          f"Image will be considered 16 bit for scaling.")
        else:
            raise ValueError(f"Invalid values in input image. They should range from 0 to 255 or 65535, not"
                             f"{min_val} to {max_val}.")
        return orig_range


    def __call__(self, sample):
        """
        Args:
            sample (ndarray): Image to be scaled.

        Returns:
            ndarray: Scaled image.
        """
        out_dtype = sample['sat_img_dtype']
        orig_range = self.range_values_raster(sample['sat_img'], out_dtype)
        sample['sat_img'] = minmax_scale(img=sample['sat_img'], orig_range=orig_range, scale_range=(self.sc_min, self.sc_max))

        return sample


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
        sample['sat_img'] = sat_img
        sample['map_img'] = map_img
        return sample


class RandomRotationTarget(object):
    """Rotate the image and target randomly."""
    def __init__(self, limit, prob, ignore_index):
        self.limit = limit
        self.prob = prob
        self.ignore_index = ignore_index

    def __call__(self, sample):
        if random.random() < self.prob:
            angle = np.random.uniform(-self.limit, self.limit)
            sat_img = transform.rotate(sample['sat_img'], angle, preserve_range=True, cval=np.nan)
            map_img = transform.rotate(sample['map_img'], angle, preserve_range=True, order=0, cval=self.ignore_index)
            sample['sat_img'] = sat_img
            sample['map_img'] = map_img
            return sample
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
            sample['sat_img'] = sat_img
            sample['map_img'] = map_img
            return sample
        else:
            return sample


class RandomCrop(object):  # TODO: what to do with overlap in samples_prep (images_to_samples, l.106)? overlap doesn't need to be larger than, say, 5%
    """Randomly crop image according to a certain dimension.
    Adapted from https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomCrop
    to support >3 band images (not currently supported by PIL)"""
    def __init__(self, sample_size, padding=3, pad_if_needed=True, ignore_index=0):
        if isinstance(sample_size, numbers.Number):
            self.size = (int(sample_size), int(sample_size))
        else:
            self.size = sample_size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.ignore_index = ignore_index

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        """
        Args:
            sample (ndarray): Image to be cropped.

        Returns:
            ndarray: Cropped image.
        """
        sat_img = sample['sat_img']
        map_img = sample['map_img']

        if self.padding is not None:
            sat_img = pad(sat_img, self.padding, np.nan)  # Pad with nan values for sat_img
            map_img = pad(map_img, self.padding, self.ignore_index)  # Pad with dontcare values for map_img

        # pad the height if needed
        if self.pad_if_needed and sat_img.shape[0] < self.size[0]:
            sat_img = pad(sat_img, (0, self.size[0] - sat_img.shape[0]), np.nan)
        # pad the width if needed
        if self.pad_if_needed and sat_img.shape[1] < self.size[1]:
            sample = pad(sat_img, (self.size[1] - sat_img.shape[1], 0), np.nan)

        # pad the height if needed
        if self.pad_if_needed and map_img.shape[0] < self.size[0]:
            map_img = pad(map_img, (0, self.size[0] - map_img.shape[0]), self.ignore_index)
        # pad the width if needed
        if self.pad_if_needed and map_img.shape[1] < self.size[1]:
            map_img = pad(map_img, (self.size[1] - map_img.shape[1], 0), self.ignore_index)

        i, j, h, w = self.get_params(sat_img, self.size)

        sat_img = sat_img[i:i + h, j:j + w]
        map_img = map_img[i:i + h, j:j + w]

        sample['sat_img'] = sat_img
        sample['map_img'] = map_img
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img = np.nan_to_num(sample['sat_img'], copy=False, nan=0.0)
        sat_img = np.float32(np.transpose(sat_img, (2, 0, 1)))
        sat_img = torch.from_numpy(sat_img)

        map_img = np.int64(sample['map_img'])
        map_img = torch.from_numpy(map_img)
        return {'sat_img': sat_img, 'map_img': map_img}
