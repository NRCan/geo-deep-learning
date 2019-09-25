import numpy as np

def minmax_scale(img, scale_range=(0, 1), orig_range=(0, 255)):
    """

    :param img: (numpy array) Image to be scaled
    :param scale_range: Desired range of transformed data.
    :param orig_range: Original range of input data.
    :return: (numpy array) Scaled image
    """
    # range(0, 1)
    scale_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    scale_img = scale_img * (scale_range[1] - scale_range[0]) + scale_range[0]
    return scale_img


def meanstd_normalize(img, mean, std=0):
    """

    :param img: (numpy array) Image to be scaled
    :param mean: (int) mean value to be subtracted
    :param std: (int) standard deviation to use as quotient
    :return: (numpy array) normalized image
    """
    mean = np.asarray(mean)
    std = np.asarray(std)
    norm_img = (img - mean) / std
    return norm_img