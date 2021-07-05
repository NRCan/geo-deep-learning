import csv
import numbers
import subprocess
from pathlib import Path
from typing import Sequence, List

import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
from torch import nn
import numpy as np
import scipy.signal
import warnings
import matplotlib
import matplotlib.pyplot as plt
import collections

from utils.readers import read_parameters

# matplotlib.use('Agg')

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


def load_from_checkpoint(checkpoint, model, optimizer=None):
    """Load weights from a previous checkpoint
    Args:
        checkpoint: (dict) checkpoint
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

def minmax_scale(img, scale_range=(0, 1), orig_range=(0, 255)):
    """
    scale data values from original range to specified range
    :param img: (numpy array) Image to be scaled
    :param scale_range: Desired range of transformed data (0, 1) or (-1, 1).
    :param orig_range: Original range of input data.
    :return: (numpy array) Scaled image
    """
    assert scale_range == (0, 1) or scale_range == (-1, 1), 'expects scale_range as (0, 1) or (-1, 1)'
    if scale_range == (0, 1):
        scale_img = (img.astype(np.float32) - orig_range[0]) / (orig_range[1] - orig_range[0])
    else:
        scale_img = 2.0 * (img.astype(np.float32) - orig_range[0]) / (orig_range[1] - orig_range[0]) - 1.0
    return scale_img

def unscale(img, float_range=(0, 1), orig_range=(0, 255)):
    """
    unscale data values from float range (0, 1) or (-1, 1) to original range (0, 255)
    :param img: (numpy array) Image to be scaled
    :param float_range: (0, 1) or (-1, 1).
    :param orig_range: (0, 255) or (0, 65535).
    :return: (numpy array) Unscaled image
    """
    f_r = float_range[1] - float_range[0]
    o_r = orig_range[1] - orig_range[0]
    return (o_r * (img - float_range[0]) / f_r) + orig_range[0]

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


def pad_diff(actual_height, actual_width, desired_height, desired_width):
    """ Pads img_arr width or height < samples_size with zeros """
    h_diff = desired_height - actual_height
    w_diff = desired_width - actual_width
    padding = (0, 0, w_diff, h_diff)  # left, top, right, bottom
    return padding


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


def ind2rgb(arr, color):
    """
    :param arr: (numpy array) index image to be color mapped
    :param color: (dict of RGB color values) for each class
    :return: (numpy_array) RGB image
    """
    h, w = arr.shape
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    for cl in color:
        for ch in range(3):
          rgb[..., ch][arr == cl] = (color[cl][ch])
    return rgb


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
        if str(img_dir_or_csv).endswith('.csv'):
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
    """
    Open csv file and parse it, returning a list of dict.
    - tif full path
    - metadata yml full path (may be empty string if unavailable)
    - gpkg full path
    - attribute_name
    - dataset (trn or tst)
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
            if row[2]:
                assert Path(row[2]).is_file(), f'Gpkg not found "{row[2]}"'
                assert isinstance(row[3], str)
    try:
        # Try sorting according to dataset name (i.e. group "train", "val" and "test" rows together)
        list_values = sorted(list_values, key=lambda k: k['dataset'])
    except TypeError:
        list_values
    return list_values


def add_metadata_from_raster_to_sample(sat_img_arr: np.ndarray,
                                       raster_handle: dict,
                                       meta_map: dict,
                                       raster_info: dict
                                       ) -> dict:
    """
    @param sat_img_arr: source image as array (opened with rasterio.read)
    @param meta_map: meta map parameter from yaml (global section)
    @param raster_info: info from raster as read with read_csv (except at inference)
    @return: Returns a metadata dictionary populated with info from source raster, including original csv line and
             histogram.
    """
    metadata_dict = {'name': raster_handle.name, 'csv_info': raster_info, 'source_raster_bincount': {}}
    assert 'dtype' in raster_handle.meta.keys(), "\"dtype\" could not be found in source image metadata"
    metadata_dict.update(raster_handle.meta)
    if not metadata_dict['dtype'] in ["uint8", "uint16"]:
        warnings.warn(f"Datatype should be \"uint8\" or \"uint16\". Got \"{metadata_dict['dtype']}\". ")
        if sat_img_arr.min() >= 0 and sat_img_arr.max() <= 255:
            metadata_dict['dtype'] = "uint8"
        elif sat_img_arr.min() >= 0 and sat_img_arr.max() <= 65535:
            metadata_dict['dtype'] = "uint16"
        else:
            raise ValueError(f"Min and max values of array ({[sat_img_arr.min(), sat_img_arr.max()]}) are not contained"
                             f"in 8 bit nor 16 bit range. Datatype cannot be overwritten.")
    # Save bin count (i.e. histogram) to metadata
    assert isinstance(sat_img_arr, np.ndarray) and len(sat_img_arr.shape) == 3, f"Array should be 3-dimensional"
    for band_index in range(sat_img_arr.shape[2]):
        band = sat_img_arr[..., band_index]
        metadata_dict['source_raster_bincount'][f'band{band_index}'] = {count for count in np.bincount(band.flatten())}
    if meta_map:
        assert raster_info['meta'] is not None and isinstance(raster_info['meta'], str) \
               and Path(raster_info['meta']).is_file(), "global configuration requested metadata mapping onto loaded " \
                                                        "samples, but raster did not have available metadata"
        yaml_metadata = read_parameters(raster_info['meta'])
        metadata_dict.update(yaml_metadata)
    return metadata_dict

#### Image Patches Smoothing Functions ####
""" Adapted from : https://github.com/Vooban/Smoothly-Blend-Image-Patches  """


def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), -1)
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    return wind


def get_git_hash():
    """
    Get git hash during execution of python script
    @return: (str) hash code for current version of geo-deep-learning. If necessary, the code associated to this hash can be
    found with the following url: https://github.com/<owner>/<project>/commit/<hash>, aka
    https://github.com/NRCan/geo-deep-learning/commit/<hash>
    """
    command = f'git rev-parse --short HEAD'
    subproc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    git_hash = str(subproc.stdout, "utf-8").replace("\n", "")
    # when code not executed from git repo, subprocess outputs return code #128. This has been tested.
    # Reference: https://stackoverflow.com/questions/58575970/subprocess-call-with-exit-status-128
    if subproc.returncode == 128:
        warnings.warn(f'No git repo associated to this code.')
        return None
    return git_hash


def ordereddict_eval(str_to_eval: str):
    """
    Small utility to successfully evaluate an ordereddict object that was converted to str by repr() function.
    @param str_to_eval: (str) string to prepared for import with eval()
    """
    # Replaces "ordereddict" string to "Collections.OrderedDict"
    if isinstance(str_to_eval, str) and "ordereddict" in str_to_eval:
        str_to_eval = str_to_eval.replace("ordereddict", "collections.OrderedDict")
        return eval(str_to_eval)
    else:
        warnings.warn(f'Object of type \"{type(str_to_eval)}\" cannot not be evaluated. Problems may occur.')
        return str_to_eval


def defaults_from_params(params, key=None):
    d = {}
    data_path = get_key_def('data_path', params['global'], '')
    preprocessing_path = get_key_def('preprocessing_path', params['global'], '')
    mlflow_experiment_name = get_key_def('mlflow_experiment_name', params['global'], 'gdl-training')
    d['prep_csv_file'] = Path(preprocessing_path, mlflow_experiment_name,
                              f"images_to_samples_{mlflow_experiment_name}.csv")
    d['img_dir_or_csv_file'] = Path(preprocessing_path, mlflow_experiment_name,
                                    f"inference_sem_seg_{mlflow_experiment_name}.csv")
    samples_size = params["global"]["samples_size"]
    overlap = params["sample"]["overlap"]
    min_annot_perc = get_key_def('min_annotated_percent', params['sample']['sampling_method'], None,
                                 expected_type=float)
    num_bands = params['global']['number_of_bands']
    d['samples_dir_name'] = (f'samples{samples_size}_overlap{overlap}_min-annot{min_annot_perc}_'
                             f'{num_bands}bands_{mlflow_experiment_name }')
    config_file_name = Path(get_key_def('config_file', params['self'], '')).stem
    d['state_dict_path'] = Path(data_path, d['samples_dir_name'], 'model', config_file_name, 'checkpoint.pth.tar')
    if key is None:
        return d
    return d[key]
