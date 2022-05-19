import csv
import logging
import numbers
import subprocess
from functools import reduce
from pathlib import Path
from typing import Sequence, List, Dict, Union

from hydra.utils import to_absolute_path
from pytorch_lightning.utilities import rank_zero_only
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf, ListConfig
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2575
import torch
from torchvision import models
import numpy as np
import scipy.signal
import requests
from urllib.parse import urlparse

# These two import statements prevent exception when using eval(metadata) in SegmentationDataset()'s __init__()
from rasterio.crs import CRS
from affine import Affine

from utils.logger import get_logger

# Set the logging file
log = get_logger(__name__)  # need to be different from logging in this case

# NVIDIA library
try:
    from pynvml import *
except ModuleNotFoundError:
    logging.warning(f"The python Nvidia management library could not be imported. Ignore if running on CPU only.")
# AWS module
try:
    import boto3
except ModuleNotFoundError:
    logging.warning('The boto3 library counldn\'t be imported. Ignore if not using AWS s3 buckets', ImportWarning)


class Interpolate(torch.nn.Module):
    def __init__(self, mode, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = torch.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


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
        logging.warning(f"No GPUs requested. This process will run on CPU")
        return lst_free_devices
    if not torch.cuda.is_available():
        log.warning(f'\nRequested {number_requested} GPUs, but no CUDA devices found. This process will run on CPU')
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


def set_device(gpu_devices_dict: dict = {}):
    """
    From dictionary of available devices, sets the device to be used
    @param gpu_devices_dict: dictionary containing info on GPU devices as returned by lst_device_ids
    @return: torch.device
    """
    if gpu_devices_dict:
        logging.info(f"\nCuda devices available: {gpu_devices_dict}.\nUsing {list(gpu_devices_dict.keys())[0]}\n\n")
        device = torch.device(f'cuda:{list(range(len(gpu_devices_dict.keys())))[0]}')
    else:
        logging.warning(f"\nNo Cuda device available. This process will only run on CPU")
        device = torch.device('cpu')
        try:
            models.resnet18().to(device)  # test with a small model
        except (RuntimeError, AssertionError):  # HPC: when device 0 not available. Error: Cuda invalid device ordinal.
            logging.warning(f"\nUnable to use device. Trying device 'cuda', not {device}")
            device = torch.device(f'cuda')
    return device


def get_key_def(key, config, default=None, expected_type=None, to_path: bool = False,
                validate_path_exists: bool = False):
    """Returns a value given a dictionary key, or the default value if it cannot be found.
    :param key: key in dictionary (e.g. generated from .yaml)
    :param config: (dict) dictionary containing keys corresponding to parameters used in script
    :param default: default value assigned if no value found with provided key
    :param expected_type: (type) type of the expected variable.
    :param to_path: (bool) if True, parameter will be converted to a pathlib.Path object (warns if cannot be converted)
    :param validate_path_exists: (bool) if True, checks if path exists (is_path must be True)
    :return:
    """
    val = default
    if not config:
        pass
    elif isinstance(key, (list, ListConfig)):
        if len(key) <= 1:  # expects list of length more than 1 to search inside a dictionary recursively
            raise ValueError("Must provide at least two valid keys to search recursively in dictionary")
        for k in key:  # iterate through items in list
            if k in config:  # if item is a key in config, check if dictionary, else set value.
                if isinstance(config[k], (dict, DictConfig)):
                    config = config[k]
                else:
                    val = config[k]
    else:
        if key not in config or config[key] is None:  # if config exists, but key not in it
            pass
        else:
            val = config[key] if config[key] != 'None' else None
            if expected_type and val is not False:
                if not isinstance(val, expected_type):
                    raise TypeError(f"{val} is of type {type(val)}, expected {expected_type}")
    if not val:  # Skips below if statements if val is None
        return val
    if to_path or validate_path_exists:
        try:
            val = Path(to_absolute_path(val))
        except TypeError:
            logging.error(f"Couldn't convert value {val} to a pathlib.Path object")
    if validate_path_exists and not val.exists():
        raise FileNotFoundError(f"Couldn't locate path: {val}.\nProvided key: {key}")
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


def list_input_images(img_dir_or_csv: Path,
                      bucket_name: str = None,
                      glob_patterns: List = None):
    """
    Create list of images from given directory or csv file.

    :param img_dir_or_csv: (str) directory containing input images or csv with list of images
    :param bucket_name: (str, optional) name of aws s3 bucket
    :param glob_patterns: (list of str) if directory is given as input (not csv),
                           these are the glob patterns that will be used to find desired images

    returns list of dictionaries where keys are "tif" and values are paths to found images. "meta" key is also added
        if input is csv and second column contains a metadata file. Then, value is path to metadata file.
    """
    if bucket_name:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        if img_dir_or_csv.suffix == '.csv':
            bucket.download_file(str(img_dir_or_csv), 'img_csv_file.csv')
            list_img = read_csv('img_csv_file.csv')
        else:
            raise NotImplementedError(
                'Specify a csv file containing images for inference. Directory input not implemented yet')
    else:
        if img_dir_or_csv.suffix == '.csv':
            list_img = read_csv(img_dir_or_csv)
        elif is_url(str(img_dir_or_csv)):
            list_img = []
            img = {'tif': img_dir_or_csv}
            list_img.append(img)
        else:
            img_dir = img_dir_or_csv
            if not img_dir.is_dir():
                raise NotADirectoryError(f'Could not find directory/file "{img_dir_or_csv}"')

            list_img_paths = set()
            if img_dir.is_dir():
                for glob_pattern in glob_patterns:
                    if not isinstance(glob_pattern, str):
                        raise TypeError(f'Invalid glob pattern: "{glob_pattern}"')
                    list_img_paths.update(sorted(img_dir.glob(glob_pattern)))
            else:
                list_img_paths.update([img_dir])
            list_img = []
            for img_path in list_img_paths:
                img = {'tif': img_path}
                list_img.append(img)
            if not len(list_img) >= 0:
                raise ValueError(f'No .tif files found in {img_dir_or_csv}')
    return list_img


def read_csv(csv_file_name: str) -> Dict:
    """
    Open csv file and parse it, returning a list of dictionaries with keys:
    - "tif": path to a single image
    - "gpkg": path to a single ground truth file
    - dataset: (str) "trn" or "tst"
    - aoi_id: (str) a string id for area of interest
    @param csv_file_name:
        path to csv file containing list of input data with expected columns
        expected columns (without header): imagery, ground truth, dataset[, aoi id]
    """
    list_values = []
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        row_lengths_set = set()
        for row in reader:
            row_lengths_set.update([len(row)])
            if not len(row_lengths_set) == 1:
                raise ValueError(f"Rows in csv should be of same length. Got rows with length: {row_lengths_set}")
            row.extend([None] * (4 - len(row)))  # fill row with None values to obtain row of length == 5
            row[0] = to_absolute_path(row[0])  # Convert relative paths to absolute with hydra's util to_absolute_path()
            if not Path(row[0]).is_file():
                logging.critical(f"Raster not found: {row[0]}. This data will be removed from input data list"
                                 f"since all geo-deep-learning modules require imagery.")
                continue
            row[1] = to_absolute_path(row[1])
            if not Path(row[1]).is_file():
                logging.critical(f"Ground truth not found: {row[1]}")
            if row[2] and not row[2] in ['trn', 'tst']:
                logging.critical(f'Invalid dataset split: {row[2]}. Expected "trn" or "tst".')
            # save all values
            list_values.append(
                {'tif': str(row[0]), 'gpkg': str(row[1]), 'dataset': row[2], 'aoi_id': row[3]})
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
        logging.warning(f"Datatype should be \"uint8\" or \"uint16\". Got \"{metadata_dict['dtype']}\". ")
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


def getpath(d, path):
    """
    TODO
    """
    return reduce(lambda acc, i: acc[i], path.split('.'), d)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "general.task",
        "mode",
        "loss",
        "dataset",
        "general.work_dir",
        "general.config_name",
        "general.config_path",
        "general.project_name",
        "general.workspace",
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
            "general.save_weights_dir",
        )
    elif config.get('mode') == 'inference':
        fields += (
            "inference",
            "model",
            "general.sample_data_dir",
        )

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


def update_gdl_checkpoint(checkpoint_params: Dict) -> Dict:
    """
    Utility to update model checkpoints from older versions of GDL to current version
    @param checkpoint_params:
        Dictionary containing weights, optimizer state and saved configuration params from training
    @return:
    """
    # covers gdl checkpoints from version <= 2.0.1
    if 'model' in checkpoint_params.keys():
        checkpoint_params['model_state_dict'] = checkpoint_params['model']
        del checkpoint_params['model']
    if 'optimizer' in checkpoint_params.keys():
        checkpoint_params['optimizer_state_dict'] = checkpoint_params['optimizer']
        del checkpoint_params['optimizer']

    # covers gdl checkpoints pre-hydra (<=2.0.0)
    bands = ['R', 'G', 'B', 'N']
    old2new = {
        'manet_pretrained': {
            '_target_': 'segmentation_models_pytorch.MAnet', 'encoder_name': 'resnext50_32x4d',
            'encoder_weights': 'imagenet'
        },
        'unet_pretrained': {
            '_target_': 'segmentation_models_pytorch.Unet', 'encoder_name': 'resnext50_32x4d',
            'encoder_depth': 4, 'encoder_weights': 'imagenet', 'decoder_channels': [256, 128, 64, 32]
        },
        'unet': {
            '_target_': 'models.unet.UNet', 'dropout': False, 'prob': False
        },
        'unet_small': {
            '_target_': 'models.unet.UNetSmall', 'dropout': False, 'prob': False
        },
        'deeplabv3_pretrained': {
            '_target_': 'segmentation_models_pytorch.DeepLabV3', 'encoder_name': 'resnet101',
            'encoder_weights': 'imagenet'
        },
        'deeplabv3_resnet101_dualhead': {
            '_target_': 'models.deeplabv3_dualhead.DeepLabV3_dualhead', 'conc_point': 'conv1',
            'encoder_weights': 'imagenet'
        },
        'deeplabv3+_pretrained': {
            '_target_': 'segmentation_models_pytorch.DeepLabV3Plus', 'encoder_name': 'resnext50_32x4d',
            'encoder_weights': 'imagenet'
        },
    }
    try:
        # don't update if already a recent checkpoint
        get_key_def('classes_dict', checkpoint_params['params']['dataset'], expected_type=(dict, DictConfig))
        get_key_def('modalities', checkpoint_params['params']['dataset'], expected_type=Sequence)
        get_key_def('model', checkpoint_params['params'], expected_type=(dict, DictConfig))
        return checkpoint_params
    except KeyError:
        num_classes_ckpt = get_key_def('num_classes', checkpoint_params['params']['global'], expected_type=int)
        num_bands_ckpt = get_key_def('number_of_bands', checkpoint_params['params']['global'], expected_type=int)
        model_name = get_key_def('model_name', checkpoint_params['params']['global'], expected_type=str)
        try:
            model_ckpt = old2new[model_name]
        except KeyError as e:
            logging.critical(f"\nCouldn't locate yaml configuration for model architecture {model_name} as found "
                             f"in provided checkpoint. Name of yaml may have changed."
                             f"\nError {type(e)}: {e}")
            raise e
        # For GDL pre-v2.0.2
        #bands_ckpt = ''
        #bands_ckpt = bands_ckpt.join([bands[i] for i in range(num_bands_ckpt)])
        checkpoint_params['params'].update({
            'dataset': {
                'modalities': [bands[i] for i in range(num_bands_ckpt)], #bands_ckpt,
                #"classes_dict": {f"BUIL": 1}
                "classes_dict": {f"class{i + 1}": i + 1 for i in range(num_classes_ckpt)}
            }
        })
        checkpoint_params['params'].update({'model': model_ckpt})
        return checkpoint_params