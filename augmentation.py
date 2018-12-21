import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import random
import numpy as np
from skimage import transform, exposure


class RandomRotationTarget(object):
    """ Rotate the image and target randomly. The rotation degree is between -90 and 90 deg."""

    def __call__(self, sample):
        angle = np.random.uniform(-90, 90)
        sat_img = transform.rotate(sample['sat_img'], angle, preserve_range=True)
        map_img = transform.rotate(sample['map_img'], angle, preserve_range=True)
        return {'sat_img': sat_img, 'map_img': map_img}


class ContrastStretching(object):
    """Rescale intensity of input image using contrast stretching.
    http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    """

    def __call__(self, sample):
        v_min, v_max = np.percentile(sample['sat_img'], (2, 98))
        sat_img = np.nan_to_num(exposure.rescale_intensity(sample['sat_img'], in_range=(v_min, v_max)))
        return {'sat_img': sat_img, 'map_img': sample['map_img']}


class HorizontalFlip(object):
    """Flip the input image and reference map horizontally, with a probability of 0.5."""

    def __call__(self, sample):
        if random.random() < 0.5:
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
