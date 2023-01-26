import logging

import numpy as np
from skimage.exposure.exposure import intensity_range

from utils.logger import get_logger

# Set the logging file
log = get_logger(__name__)  # need to be different from logging in this case


def stretch_heatmap(heatmap_arr: np.ndarray, out_max: int = 100, range_warning: bool = True) -> np.ndarray:
    """
    Stretches heatmap values between 0 and an inputted maximum value
    @param heatmap_arr:
        3D array of dtype float containing probability map for each class,
        after sigmoid or softmax operation (expects values between 0 and 1)
    @param out_max:
        Output maximum value
    @param range_warning:
        if True, a warning will be emitted if values of heatmap range under 0 or above 1.
    @return: numpy array with stretched values
    """
    imin, imax = map(float, intensity_range(heatmap_arr, 'image'))
    if imin < 0 or imax > 1:
        if range_warning:
            logging.warning(f"\nProvided heatmap should be the result of sigmoid or softmax operation."
                            f"\nExpected values are between 0 and 1. Got min {imin} and max {imax}.")
        dtype_max = imax
    else:
        _, dtype_max = map(float, intensity_range(heatmap_arr, 'dtype'))
    return np.array(heatmap_arr) / dtype_max * out_max


def class_from_heatmap(heatmap_arr: np.ndarray, heatmap_threshold: float = 0.5) -> np.ndarray:
    """
    Sets class value from raw heatmap as predicted by model
    @param heatmap_arr:
        3D array (channels last) of dtype float containing probability map for each class,
        after sigmoid or softmax operation (expects values between 0 and 1)
    @param heatmap_threshold:
        threshold (fraction of 1) to apply to heatmap if single class prediction
    @return: flattened array where pixel values correspond to final class values
    """
    if heatmap_arr.shape[-1] == 1:
        heatmap_threshold_abs = heatmap_threshold * np.iinfo(heatmap_arr.dtype).max
        flattened_arr = (heatmap_arr > heatmap_threshold_abs)
        flattened_arr = np.squeeze(flattened_arr, axis=-1)
    else:
        flattened_arr = heatmap_arr.argmax(axis=-1)
    return flattened_arr.astype(np.uint8)
