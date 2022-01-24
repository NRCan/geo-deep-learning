import os
import csv
import logging
import numbers
import subprocess
import importlib as imp
from functools import reduce
from pathlib import Path
from typing import Sequence, List
from pytorch_lightning.utilities import rank_zero_only

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
from torch import nn
import numpy as np
import scipy.signal
import warnings
import requests
import collections

# These two import statements prevent exception when using eval(metadata) in SegmentationDataset()'s __init__()
from rasterio.crs import CRS
from affine import Affine

from utils.readers import read_parameters
from urllib.parse import urlparse

try:
    from ruamel_yaml import YAML
except ImportError:
    from ruamel.yaml import YAML
# NVIDIA library
try:
    from pynvml import *
except ModuleNotFoundError:
    warnings.warn(f"The python Nvidia management library could not be imported. Ignore if running on CPU only.")
# AWS module
try:
    import boto3
except ModuleNotFoundError:
    warnings.warn('The boto3 library counldn\'t be imported. Ignore if not using AWS s3 buckets', ImportWarning)
    pass


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


# Set the logging file
log = get_logger(__name__)  # need to be different from logging in this case


class Interpolate(torch.nn.Module):
    def __init__(self, mode, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = torch.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


def load_from_checkpoint(checkpoint, model, optimizer=None, inference: str = ''):
    """Load weights from a previous checkpoint
    Args:
        checkpoint: (dict) checkpoint
        model: model to replace
        optimizer: optimiser to be used
        inference: (str) path to inference state_dict. If given, loading will be strict (see pytorch doc)
    """
    # Corrects exception with test loop. Problem with loading generic checkpoint into DataParallel model
    # model.load_state_dict(checkpoint['model'])
    # https://github.com/bearpaw/pytorch-classification/issues/27
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    if isinstance(model, nn.DataParallel) and not list(checkpoint['model'].keys())[0].startswith('module'):
        new_state_dict = model.state_dict().copy()
        new_state_dict['model'] = {'module.'+k: v for k, v in checkpoint['model'].items()}    # Very flimsy
        del checkpoint
        checkpoint = {}
        checkpoint['model'] = new_state_dict['model']

    strict_loading = False if not inference else True
    model.load_state_dict(checkpoint['model'], strict=strict_loading)
    log.info(f"\n=> loaded model")
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


def get_device_ids(
        number_requested: int,
        max_used_ram_perc: int = 25,
        max_used_perc: int = 15):
    """
    Function to check which GPU devices are available and unused.
    :param number_requested: (int) Number of devices requested.
    :param max_used_ram_perc: (int) If RAM usage of detected GPU exceeds this percentage, it will be ignored
    :param max_used_perc: (int) If GPU's usage exceeds this percentage, it will be ignored
    :return: (list) Unused GPU devices.
    """
    lst_free_devices = {}
    if not number_requested:
        log.warning(f'\nNo GPUs requested. This training will run on CPU')
        return lst_free_devices
    if not torch.cuda.is_available():
        log.warning(f'\nRequested {number_requested} GPUs, but no CUDA devices found. This training will run on CPU')
        return lst_free_devices
    try:
        nvmlInit()
        if number_requested > 0:
            device_count = nvmlDeviceGetCount()
            for i in range(device_count):
                res, mem = gpu_stats(i)
                used_ram = mem.used / (1024 ** 2)
                max_ram = mem.total / (1024 ** 2)
                used_ram_perc = used_ram / max_ram * 100
                log.info(f'\nGPU RAM used: {used_ram_perc} ({used_ram:.0f}/{max_ram:.0f} MiB)\nGPU % used: {res.gpu}')
                if used_ram_perc < max_used_ram_perc:
                    if res.gpu < max_used_perc:
                        lst_free_devices[i] = {'used_ram_at_init': used_ram, 'max_ram': max_ram}
                    else:
                        log.warning(f'\nGpu #{i} filtered out based on usage % threshold.\n'
                                    f'Current % usage: {res.gpu}\n'
                                    f'Max % usage allowed by user: {max_used_perc}.')
                else:
                    log.warning(f'\nGpu #{i} filtered out based on RAM threshold.\n'
                                f'Current RAM usage: {used_ram}/{max_ram}\n'
                                f'Max used RAM allowed by user: {max_used_ram_perc}.')
                if len(lst_free_devices.keys()) == number_requested:
                    break
            if len(lst_free_devices.keys()) < number_requested:
                log.warning(f"\nYou requested {number_requested} devices. {device_count} devices are available and "
                            f"other processes are using {device_count-len(lst_free_devices.keys())} device(s).")
        else:
            log.warning('\nNo gpu devices requested. Will run on cpu')
            return lst_free_devices
    except NameError as error:
        raise log.critical(
            NameError(f"\n{error}. Make sure that the NVIDIA management library (pynvml) is installed and running.")
        )
    except NVMLError as error:
        raise log.critical(
            ValueError(f"\n{error}. Make sure that the latest NVIDIA driver is installed and running.")
        )
    logging.info(f'\nGPUs devices available: {lst_free_devices}')
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
    :param expected_type: (type) type of the expected variable.
    :return:
    """
    if not config:
        return default
    elif isinstance(key, list):  # is key a list?
        if len(key) <= 1:  # is list of length 1 or shorter? else --> default
            if msg is not None:
                raise log.critical(AssertionError(msg))
            else:
                raise log.critical(AssertionError("Must provide at least two valid keys to test"))
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
            if expected_type and val is not False:
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


def is_url(url):
    if urlparse(url).scheme in ('http', 'https', 's3'):
        return True
    else:
        return False


def checkpoint_url_download(url: str):
    mime_type = ('application/tar', 'application/x-tar', 'applicaton/x-gtar',
                 'multipart/x-tar', 'application/x-compress', 'application/x-compressed')
    try:
        response = requests.head(url)
        if response.headers['content-type'] in mime_type:
            working_folder = Path.cwd().joinpath('inference_out')
            Path.mkdir(working_folder, parents=True, exist_ok=True)
            checkpoint_path = working_folder.joinpath(Path(url).name)
            r = requests.get(url)
            checkpoint_path.write_bytes(r.content)
            print(checkpoint_path)
            return checkpoint_path
        else:
            raise SystemExit('Invalid Url, checkpoint content not detected')

    except requests.exceptions.RequestException as e:
        raise SystemExit(e)


def list_input_images(img_dir_or_csv: str,
                      bucket_name: str = None,
                      glob_patterns: List = None,
                      in_case_of_path: str = None):
    """
    Create list of images from given directory or csv file.

    :param img_dir_or_csv: (str) directory containing input images or csv with list of images
    :param bucket_name: (str, optional) name of aws s3 bucket
    :param glob_patterns: (list of str) if directory is given as input (not csv),
                           these are the glob patterns that will be used to find desired images
    :param in_case_of_path: (str) directory that can contain the images if not the good one in the csv

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
            list_img = read_csv(img_dir_or_csv, in_case_of_path)
        elif is_url(img_dir_or_csv):
            list_img = []
            img_path = Path(img_dir_or_csv)
            img = {}
            img['tif'] = img_path
            list_img.append(img)
        else:
            img_dir = Path(img_dir_or_csv)
            assert img_dir.is_dir() or img_dir.is_file(), f'Could not find directory/file "{img_dir_or_csv}"'

            list_img_paths = set()
            if img_dir.is_dir():
                for glob_pattern in glob_patterns:
                    assert isinstance(glob_pattern, str), f'Invalid glob pattern: "{glob_pattern}"'
                    list_img_paths.update(sorted(img_dir.glob(glob_pattern)))
            else:
                list_img_paths.update([img_dir])
            list_img = []
            for img_path in list_img_paths:
                img = {'tif': img_path}
                list_img.append(img)
            assert len(list_img) >= 0, f'No .tif files found in {img_dir_or_csv}'
    return list_img


def try2read_csv(path_file, in_case_of_path, msg):
    """
    TODO
    """
    try:
        Path(path_file).resolve(strict=True)
    except FileNotFoundError:
        if in_case_of_path:
            path_file = os.path.join(in_case_of_path, os.path.basename(path_file))
            try:
                Path(path_file).resolve(strict=True)
            except FileNotFoundError:
                raise log.critical(f'\n{msg} "{path_file}"')
        else:
            raise log.critical(f'\n{msg} "{path_file}"')
    return path_file


def read_csv(csv_file_name, data_path=None):
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
            # verify if the path is correct, change it if not and raise error msg if not existing
            row[0] = try2read_csv(row[0], data_path, 'Tif raster not found:')
            if row[2]:
                row[2] = try2read_csv(row[2], data_path, 'Gpkg not found:')
                assert isinstance(row[3], str)
            # save all values
            list_values.append(
                {'tif': row[0], 'meta': row[1], 'gpkg': row[2], 'attribute_name': row[3], 'dataset': row[4]}
            )
    try:
        # Try sorting according to dataset name (i.e. group "train", "val" and "test" rows together)
        list_values = sorted(list_values, key=lambda k: k['dataset'])
    except TypeError:
        log.warning('Unable to sort csv rows')
    return list_values


def add_metadata_from_raster_to_sample(sat_img_arr: np.ndarray,
                                       raster_handle: dict,
                                       raster_info: dict
                                       ) -> dict:
    """
    :param sat_img_arr: source image as array (opened with rasterio.read)
    :param raster_info: info from raster as read with read_csv (except at inference)
    :return: Returns a metadata dictionary populated with info from source raster, including original csv line and
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
            raise NotImplementedError(f"Min and max values of array ({[sat_img_arr.min(), sat_img_arr.max()]}) "
                                      f"are not contained in 8 bit nor 16 bit range. Datatype cannot be overwritten.")
    # Save bin count (i.e. histogram) to metadata
    assert isinstance(sat_img_arr, np.ndarray) and len(sat_img_arr.shape) == 3, f"Array should be 3-dimensional"
    for band_index in range(sat_img_arr.shape[2]):
        band = sat_img_arr[..., band_index]
        metadata_dict['source_raster_bincount'][f'band{band_index}'] = {count for count in np.bincount(band.flatten())}
    return metadata_dict

#### Image Patches Smoothing Functions ####
""" Adapted from : https://github.com/Vooban/Smoothly-Blend-Image-Patches  """


def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2 * (scipy.signal.windows.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.windows.triang(window_size) - 1)) ** power) / 2
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
    :return: (str) hash code for current version of geo-deep-learning. If necessary, the code associated to this hash can be
    found with the following url: https://github.com/<owner>/<project>/commit/<hash>, aka
    https://github.com/NRCan/geo-deep-learning/commit/<hash>
    """
    command = f'git rev-parse --short HEAD'
    subproc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    git_hash = str(subproc.stdout, "utf-8").replace("\n", "")
    # when code not executed from git repo, subprocess outputs return code #128. This has been tested.
    # Reference: https://stackoverflow.com/questions/58575970/subprocess-call-with-exit-status-128
    if subproc.returncode == 128:
        log.warning(f'No git repo associated to this code.')
        return None
    return git_hash


def ordereddict_eval(str_to_eval: str):
    """
    Small utility to successfully evaluate an ordereddict object that was converted to str by repr() function.
    :param str_to_eval: (str) string to prepared for import with eval()
    """
    try:
        # Replaces "ordereddict" string to "Collections.OrderedDict"
        if isinstance(str_to_eval, bytes):
            str_to_eval = str_to_eval.decode('UTF-8')
        str_to_eval = str_to_eval.replace("ordereddict", "collections.OrderedDict")
        return eval(str_to_eval)
    except Exception:
        log.exception(f'Object of type \"{type(str_to_eval)}\" cannot not be evaluated. Problems may occur.')
        return str_to_eval


def compare_config_yamls(yaml1: dict, yaml2: dict, update_yaml1: bool = False) -> List:
    """
    Checks if values for same keys or subkeys (max depth of 2) of two dictionaries match.
    :param yaml1: (dict) first dict to evaluate
    :param yaml2: (dict) second dict to evaluate
    :param update_yaml1: (bool) it True, values in yaml1 will be replaced with values in yaml2,
                         if the latters are different
    :return: dictionary of keys or subkeys for which there is a value mismatch if there is, or else returns None
    """
    # TODO need to be change if the training or testing config are not the same as when the sampling have been create
    # TODO maybe only check and save a small part of the config like the model or something

    if not (isinstance(yaml1, dict) or isinstance(yaml2, dict)):
        raise TypeError(f"\nExpected both yamls to be dictionaries. \n"
                        f"Yaml1's type is  {type(yaml1)}\n"
                        f"Yaml2's type is  {type(yaml2)}")
    for section, params in yaml2.items():  # loop through main sections of config yaml ('global', 'sample', etc.)
        if section in {'task', 'mode', 'debug'}:  # the task is not the same as the hdf5 since the hdf5 is in sampling
            continue
        if section not in yaml1.keys():  # create key if not in dictionary as we loop
            yaml1[section] = {}
        for param, val2 in params.items():  # loop through parameters of each section ('samples_size','debug_mode',...)
            if param in {'config_override_dirname'}:  # the config_override_dirname is not the same as the hdf5 since the hdf5 is in sampling
                continue
            if param not in yaml1[section].keys():  # create key if not in dictionary as we loop
                yaml1[section][param] = {}
            # set to None if no value for that key
            val1 = get_key_def(param, yaml1[section], default=None)
            if isinstance(val2, dict):  # if value is a dict, loop again to fetch end val (only recursive twice)
                for subparam, subval2 in val2.items():
                    if subparam not in yaml1[section][param].keys():  # create key if not in dictionary as we loop
                        yaml1[section][param][subparam] = {}
                    # set to None if no value for that key
                    subval1 = get_key_def(subparam, yaml1[section][param], default=None)
                    if subval2 != subval1:
                        # if value doesn't match between yamls, emit warning
                        log.warning(f"\nYAML value mismatch: section \"{section}\", key \"{param}/{subparam}\"\n"
                                        f"Current yaml value: \"{subval1}\"\nHDF5s yaml value: \"{subval2}\"\n")
                        if update_yaml1:  # update yaml1 with subvalue of yaml2
                            yaml1[section][param][subparam] = subval2
                            log.info(f'Value in yaml1 updated')
            elif val2 != val1:
                log.warning(f"\nYAML value mismatch: section \"{section}\", key \"{param}\"\n"
                                f"Current yaml value: \"{val2}\"\nHDF5s yaml value: \"{val1}\"\n"
                                f"Problems may occur.")
                if update_yaml1:  # update yaml1 with value of yaml2
                    yaml1[section][param] = val2
                    log.info(f'Value in yaml1 updated')


def load_obj(obj_path: str, default_obj_path: str = '') -> any:
    """
    Extract an object from a given path.

    :param obj_path: (str) Path to an object to be extracted, including the object name.
    :param default_obj_path: (str) Default path object.

    :return: Extract object. Can be a function or a class or ...

    :raise AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = imp.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def read_modalities(modalities: str) -> list:
    """
    Function that read the modalities from the yaml and convert it to a list
    of all the bands specified.

    -------
    :param modalities: (str) A string composed of all the bands of the images.

    -------
    :returns: A list of all the bands of the images.
    """
    if str(modalities).find('IR') != -1:
        ir_position = str(modalities).find('IR')
        modalities = list(str(modalities).replace('IR', ''))
        modalities.insert(ir_position, 'IR')
    else:
        modalities = list(str(modalities))
    return modalities


def find_first_file(name, list_path):
    """
    TODO
    """
    for dirname in list_path:
        # print("dir:", dirname)
        for root, dirs, files in os.walk(os.path.dirname(dirname)):
            # print(root, dirs, files)
            for filename in files:
                # print("file:", filename)
                if filename == name:
                    return dirname
                    # return os.path.join(dirname, name)


def getpath(d, path):
    """
    TODO
    """
    return reduce(lambda acc, i: acc[i], path.split('.'), d)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "task",
        "mode",
        "dataset",
        "general.work_dir",
        "general.config_name",
        "general.config_path",
        "general.project_name",
        "general.workspace",
        "general.device",
    ),
    resolve: bool = True,
) -> None:
    """
    Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    save_git_hash = tree.add('Git hash', style=style, guide_style=style)
    save_git_hash.add(str(getpath(config, 'general.git_hash')))
    save_dir = tree.add('Saving directory', style=style, guide_style=style)
    save_dir.add(os.getcwd())

    if config.get('mode') == 'sampling':
        fields += (
            "general.raw_data_dir",
            "general.raw_data_csv",
            "general.sample_data_dir",
        )
    elif config.get('mode') == 'train':
        fields += (
            "model",
            "training",
            'optimizer',
            'callbacks',
            'scheduler',
            'augmentation',
            "general.sample_data_dir",
            "general.state_dict_path",
            "general.save_weights_dir",
        )
    elif config.get('mode') == 'inference':
        fields += (
            "model",
            "general.sample_data_dir",
            "general.state_dict_path",
        )

    if getpath(config, 'AWS.bucket_name'):
        fields += ("AWS",)

    if config.get('tracker'):
        fields += ("tracker",)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        # config_section = config.get(field)
        config_section = getpath(config, field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, "yaml", word_wrap=True))

    if config.get('debug'):
        rich.print(tree, flush=False)

    with open("run_config.config", "w") as fp:
        rich.print(tree, file=fp)


# def save_useful_info():
#     shutil.copytree(
#         os.path.join(hydra.utils.get_original_cwd(), 'src'),
#         os.path.join(os.getcwd(), 'code/src')
#     )
#     shutil.copy2(
#         os.path.join(hydra.utils.get_original_cwd(), 'hydra_run.py'),
#         os.path.join(os.getcwd(), 'code')
#     )
