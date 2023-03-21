import logging

import numpy as np
import scipy.signal.windows as w
from skimage.exposure.exposure import intensity_range
from typing import List

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


def window2d(window_size: int) -> np.ndarray:
    """
    Returns a 2D square signal image generated from hann window.

    Args:
        window_size (int): Same as Chunk/Tile size used at inference.

    Returns:
        ndarray (2D): Signal window with values ranging from 0 to 1.
    """
    window = np.matrix(w.hann(M=window_size, sym=False))
    return window.T.dot(window)


def generate_corner_windows(window_size: int) -> np.ndarray:
    """
    Generates 9 2D signal windows that covers edge and corner coordinates
    
    Args:
        window_size:

    Returns:
        ndarray: 9 2D signal windows stacked in array (3, 3)

    """
    step = window_size >> 1
    window = window2d(window_size)
    window_u = np.vstack([np.tile(window[step:step+1, :], (step, 1)), window[step:, :]])
    window_b = np.vstack([window[:step, :], np.tile(window[step:step+1, :], (step, 1))])
    window_l = np.hstack([np.tile(window[:, step:step+1], (1, step)), window[:, step:]])
    window_r = np.hstack([window[:, :step], np.tile(window[:, step:step+1], (1, step))])
    window_ul = np.block([[np.ones((step, step)), window_u[:step, step:]],
                          [window_l[step:, :step], window_l[step:, step:]]])
    window_ur = np.block([[window_u[:step, :step], np.ones((step, step))],
                          [window_r[step:, :step], window_r[step:, step:]]])
    window_bl = np.block([[window_l[:step, :step], window_l[:step, step:]],
                          [np.ones((step, step)), window_b[step:, step:]]])
    window_br = np.block([[window_r[:step, :step], window_r[:step, step:]],
                          [window_b[step:, :step], np.ones((step, step))]])
    return np.array([[window_ul, window_u, window_ur],
                     [window_l, window, window_r],
                     [window_bl, window_b, window_br]])


def generate_patch_list(image_width: int, image_height: int, window_size: int, overlapping: bool=False)-> List[tuple]:
    """
    Generates a list of patches from an image with given width, height and window size
    The patches can be generated with overlapping or non-overlapping windows
    Args:
        image_width (int): Width dimension of input image
        image_height (int): Height dimension of input image
        window_size (int): Size of the window used to generate patches
        overlapping (bool): Boolean parameter to generate overlaps or non-overlaps patches

    Returns:
        list: List of overlap or non-overlap patches, position and size

    """
    patch_list = []
    if overlapping:
        step = window_size >> 1
        windows = generate_corner_windows(window_size)
        max_height = int(image_height/step - 1)*step
        max_width = int(image_width/step - 1)*step
    else:
        step = window_size
        windows = np.ones((window_size, window_size))
        max_height = int(image_height/step)*step
        max_width = int(image_width/step)*step
    for i in range(0, max_height, step):
        for j in range(0, max_width, step):
            if overlapping:
                # Close to border and corner cases
                # Default (1, 1) is regular center window
                border_x, border_y = 1, 1
                if i == 0: border_x = 0
                if j == 0: border_y = 0
                if i == max_height-step: border_x = 2
                if j == max_width-step: border_y = 2
                # Selecting the right window
                current_window = windows[border_x, border_y]
            else:
                current_window = windows
            # The patch is cropped when the patch size is not
            # a multiple of the image size.
            patch_height = window_size
            if i+patch_height > image_height:
                patch_height = image_height - i
            patch_width = window_size
            if j+patch_width > image_width:
                patch_width = image_width - j
            # Adding the patch
            patch_list.append((j, i, patch_width, patch_height, current_window[:patch_height, :patch_width]))
    return patch_list

