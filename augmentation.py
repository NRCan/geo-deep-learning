from skimage import transform
import torch
from torchvision import transforms

import numpy as np

class RescaleTarget(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, tuple):
            self.output_size = int(np.random.uniform(output_size[0], output_size[1]))
        else:
            self.output_size = output_size

    def __call__(self, sample):
        sat_img, map_img = sample['sat_img'], sample['map_img']

        h, w = sat_img.shape[:2]

        if h > w:
            new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size, self.output_size * w / h

        new_h, new_w = int(new_h), int(new_w)

        # change the range to 0-1 rather than 0-255, makes it easier to use sigmoid later
        sat_img = transform.resize(sat_img, (new_h, new_w))

        map_img = transform.resize(map_img, (new_h, new_w))

        return {'sat_img': sat_img, 'map_img': map_img}


class RandomRotationSamples(object):
    """ Rotate the image and target randomly in a sample.
    Args:
        degrees (tuple or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resize (boolean): Expand the image to fit
    """

    def __init__(self, degrees, resize=False):
        if isinstance(degrees, int):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if isinstance(degrees, tuple):
                raise ValueError("Degrees needs to be either an int or tuple")
            self.degrees = degrees

        assert isinstance(resize, bool)

        self.resize = resize
        self.angle = np.random.uniform(self.degrees[0], self.degrees[1])

    def __call__(self, sample):

        sat_img = transform.rotate(sample['sat_img'], self.angle, self.resize)
        map_img = transform.rotate(sample['map_img'], self.angle, self.resize)

        return {'sat_img': sat_img, 'map_img': map_img}


class RandomCropTarget(object):
    """
    Crop the image and target randomly in a sample.
    Args:
    output_size (tuple or int): Desired output size. If int, square crop
        is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        sat_img, map_img = sample['sat_img'], sample['map_img']

        h, w = sat_img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sat_img = sat_img[top: top + new_h, left: left + new_w]
        map_img = map_img[top: top + new_h, left: left + new_w]

        return {'sat_img': sat_img, 'map_img': map_img}


class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """https://discuss.pytorch.org/t/torchvision-transforms-totensor-img-not-working/12663"""
        sat_img = transforms.ToTensor()(sample['sat_img'])
        return {'sat_img': sat_img, 'map_img': torch.from_numpy(sample['map_img'])}


class NormalizeRGB(object):
    """Normalize a tensor and also return the target"""

    def __call__(self, sample):

        # return {'sat_img': transforms.functional.normalize(sample['sat_img'], self.mean, self.std),'map_img': sample['map_img']}
        return {'sat_img': transforms.functional.normalize(sample['sat_img'], sample['sat_img'].mean(dim=0), sample['sat_img'].std(dim=0)),'map_img': sample['map_img']}

# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
