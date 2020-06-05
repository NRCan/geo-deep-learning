import csv
import numbers
from pathlib import Path
from typing import Sequence, List

import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
from torch import nn
import numpy as np
import warnings
import collections
import matplotlib

matplotlib.use('Agg')

try:
    from ruamel_yaml import YAML
except ImportError:
    from ruamel.yaml import YAML

try:
    from pynvml import *
except ModuleNotFoundError:
    warnings.warn(f"The python Nvidia management library could not be imported. Ignore if running on CPU only.")

try:
    import boto3
except ModuleNotFoundError:
    warnings.warn('The boto3 library counldn\'t be imported. Ignore if not using AWS s3 buckets', ImportWarning)
    pass


class Interpolate(torch.nn.Module):
    def __init__(self, mode, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = torch.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


def load_from_checkpoint(checkpoint, model, optimizer=None, inference=False):
    """Load weights from a previous checkpoint
    Args:
        checkpoint: (dict) checkpoint as loaded in model_choice.py
        model: model to replace
        optimizer: optimiser to be used
    """
    # Corrects exception with test loop. Problem with loading generic checkpoint into DataParallel model	    model.load_state_dict(checkpoint['model'])
    # https://github.com/bearpaw/pytorch-classification/issues/27
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    if isinstance(model, nn.DataParallel) and not list(checkpoint['model'].keys())[0].startswith('module'):
        new_state_dict = model.state_dict().copy()
        new_state_dict['model'] = {'module.'+k: v for k, v in checkpoint['model'].items()}    # Very flimsy
        del checkpoint
        checkpoint = {}
        checkpoint['model'] = new_state_dict['model']

    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"=> loaded model\n")
    if optimizer and 'optimizer' in checkpoint.keys():    # 2nd condition if loading a model without optimizer
        optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
    return model, optimizer


def list_s3_subfolders(bucket, data_path):
    list_classes = []

    client = boto3.client('s3')
    result = client.list_objects(Bucket=bucket, Prefix=data_path+'/', Delimiter='/')
    for p in result.get('CommonPrefixes'):
        if p['Prefix'].split('/')[-2] is not data_path:
            list_classes.append(p['Prefix'].split('/')[-2])
    return list_classes


def get_device_ids(number_requested, max_used_ram=2000, max_used_perc=15, debug=False):
    """
    Function to check which GPU devices are available and unused.
    :param number_requested: (int) Number of devices requested.
    :return: (list) Unused GPU devices.
    """
    lst_free_devices = []
    try:
        nvmlInit()
        if number_requested > 0:
            device_count = nvmlDeviceGetCount()
            for i in range(device_count):
                res, mem = gpu_stats(i)
                if debug:
                    print(f'GPU RAM used: {round(mem.used/(1024**2), 1)} | GPU % used: {res.gpu}')
                if round(mem.used/(1024**2), 1) <  max_used_ram and res.gpu < max_used_perc:
                    lst_free_devices.append(i)
                if len(lst_free_devices) == number_requested:
                    break
            if len(lst_free_devices) < number_requested:
                warnings.warn(f"You requested {number_requested} devices. {device_count} devices are available on this computer and "
                              f"other processes are using {device_count-len(lst_free_devices)} device(s).")
    except NameError as error:
        raise NameError(f"{error}. Make sure that the NVIDIA management library (pynvml) is installed and running.")
    except NVMLError as error:
        raise ValueError(f"{error}. Make sure that the latest NVIDIA driver is installed and running.")

    return lst_free_devices


def gpu_stats(device=0):
    """
    Provides GPU utilization (%) and RAM usage
    :return: res.gpu, res.memory
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device)
    res = nvmlDeviceGetUtilizationRates(handle)
    mem = nvmlDeviceGetMemoryInfo(handle)

    return res, mem


def get_key_def(key, config, default=None, msg=None, delete=False, expected_type=None):
    """Returns a value given a dictionary key, or the default value if it cannot be found.
    :param key: key in dictionary (e.g. generated from .yaml)
    :param config: (dict) dictionary containing keys corresponding to parameters used in script
    :param default: default value assigned if no value found with provided key
    :param msg: message returned with AssertionError si length of key is smaller or equal to 1
    :param delete: (bool) if True, deletes parameter, e.g. for one-time use.
    :return:
    """
    if not config:
        return default
    elif isinstance(key, list):  # is key a list?
        if len(key) <= 1:  # is list of length 1 or shorter? else --> default
            if msg is not None:
                raise AssertionError(msg)
            else:
                raise AssertionError("Must provide at least two valid keys to test")
        for k in key:  # iterate through items in list
            if k in config:  # if item is a key in config, set value.
                val = config[k]
                if delete:  # optionally delete parameter after defining a variable with it
                    del config[k]
        val = default
    else:  # if key is not a list
        if key not in config or config[key] is None:  # if key not in config dict
            val = default
        else:
            val = config[key] if config[key] != 'None' else None
            if expected_type:
                assert isinstance(val, expected_type), f"{val} is of type {type(val)}, expected {expected_type}"
            if delete:
                del config[key]
    return val


def get_key_recursive(key, config):
    """Returns a value recursively given a dictionary key that may contain multiple subkeys."""
    if not isinstance(key, list):
        key = key.split("/")  # subdict indexing split using slash
    assert key[0] in config, f"missing key '{key[0]}' in metadata dictionary: {config}"
    val = config[key[0]]
    if isinstance(val, (dict, collections.OrderedDict)):
        assert len(key) > 1, "missing keys to index metadata subdictionaries"
        return get_key_recursive(key[1:], val)
    return val


def minmax_scale(img, scale_range=(0, 1), orig_range=(0, 255)):
    """Rescale data values from original range to specified range

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


def pad(img, padding, fill=0):
    r"""Pad the given ndarray on all sides with specified padding mode and fill value.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py#L255
    Args:
        img (ndarray): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
    Returns:
        ndarray: Padded image.
    """
    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    # RGB image
    if len(img.shape) == 3:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=fill)
    # Grayscale image
    elif len(img.shape) == 2:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=fill)

    return img


def pad_diff(actual_height, actual_width, desired_shape):
    """ Pads img_arr width or height < samples_size with zeros """
    h_diff = desired_shape - actual_height
    w_diff = desired_shape - actual_width

    return h_diff, w_diff


def unnormalize(input_img, mean, std):

    """
    :param input_img: (numpy array) Image to be "unnormalized"
    :param mean: (list of mean values) for each channel
    :param std:  (list of std values) for each channel
    :return: (numpy_array) "Unnormalized" image
    """
    return (input_img * std) + mean


def BGR_to_RGB(array):
    assert array.shape[2] >= 3, f"Not enough channels in array of shape {array.shape}"
    BGR_channels = array[..., :3]
    RGB_channels = np.ascontiguousarray(BGR_channels[..., ::-1])
    array[:, :, :3] = RGB_channels
    return array


def list_input_images(img_dir_or_csv: str,
                      bucket_name: str = None,
                      glob_patterns: List = None):
    """
    Create list of images from given directory or csv file.

    :param img_dir_or_csv: (str) directory containing input images or csv with list of images
    :param bucket_name: (str, optional) name of aws s3 bucket
    :param glob_patterns: (list of str) if directory is given as input (not csv), these are the glob patterns that will be used
                        to find desired images

    returns list of dictionaries where keys are "tif" and values are paths to found images. "meta" key is also added
        if input is csv and second column contains a metadata file. Then, value is path to metadata file.
    """
    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        if img_dir_or_csv.endswith('.csv'):
            bucket.download_file(img_dir_or_csv, 'img_csv_file.csv')
            list_img = read_csv('img_csv_file.csv')
        else:
            raise NotImplementedError(
                'Specify a csv file containing images for inference. Directory input not implemented yet')
    else:
        if img_dir_or_csv.endswith('.csv'):
            list_img = read_csv(img_dir_or_csv)
        else:
            img_dir = Path(img_dir_or_csv)
            assert img_dir.is_dir(), f'Could not find directory "{img_dir_or_csv}"'

            list_img_paths = set()
            for glob_pattern in glob_patterns:
                assert isinstance(glob_pattern, str), f'Invalid glob pattern: "{glob_pattern}"'
                list_img_paths.update(sorted(img_dir.glob(glob_pattern)))

            list_img = []
            for img_path in list_img_paths:
                img = {}
                img['tif'] = img_path
                list_img.append(img)
            assert len(list_img) >= 0, f'No .tif files found in {img_dir_or_csv}'
    return list_img


def read_csv(csv_file_name):
    """Open csv file and parse it, returning a list of dict.

    If inference == True, the dict contains this info:
        - tif full path
        - metadata yml full path (may be empty string if unavailable)
    Else, the returned list contains a dict with this info:
        - tif full path
        - metadata yml full path (may be empty string if unavailable)
        - gpkg full path
        - attribute_name
        - dataset (trn or val)
    """

    list_values = []
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            row_length = len(row) if index == 0 else row_length
            assert len(row) == row_length, "Rows in csv should be of same length"
            row.extend([None] * (5 - len(row)))  # fill row with None values to obtain row of length == 5
            list_values.append({'tif': row[0], 'meta': row[1], 'gpkg': row[2], 'attribute_name': row[3], 'dataset': row[4]})
            assert Path(row[0]).is_file(), f'Tif raster not found "{row[0]}"'
            if row[2] is not None:
                assert Path(row[2]).is_file(), f'Gpkg not found "{row[2]}"'
                assert isinstance(row[3], str)
    try:
        # Try sorting according to dataset name (i.e. group "train", "val" and "test" rows together)
        list_values = sorted(list_values, key=lambda k: k['dataset'])
    except TypeError:
        list_values
    return list_values