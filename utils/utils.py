import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import os
from torch import nn
import numpy as np
import rasterio
import rasterio.features
import warnings
import collections
import fiona
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, gridspec
import math

from utils.preprocess import minmax_scale

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


def create_or_empty_folder(folder, empty=True):
    """Empty an existing folder or create it if it doesn't exist.
    Args:
        folder: full file path of the folder to be emptied/created
    """
    if not Path(folder).is_dir():
        Path.mkdir(Path(folder), exist_ok=False)
    elif empty is True:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)


def read_parameters(param_file):
    """Read and return parameters in .yaml file
    Args:
        param_file: Full file path of the parameters file
    Returns:
        YAML (Ruamel) CommentedMap dict-like object
    """
    yaml = YAML()
    with open(param_file) as yamlfile:
        params = yaml.load(yamlfile)
    return params


def chop_layer(pretrained_dict,
               layer_names=["logits"]):  # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    """
    Removes keys from a layer in state dictionary of model architecture.
    :param model: (nn.Module) model with original architecture
    :param layer_names: (list) names of layers to be chopped.
    :return: (nn.Module) model
    """
    # filter out weights from undesired keys. ex.: size mismatch.
    for layer in layer_names:
        chopped_dict = {k: v for k, v in pretrained_dict.items() if k.find(layer) == -1}
        pretrained_dict = chopped_dict  # overwrite values in pretrained dict with chopped dict
    return chopped_dict


def load_from_checkpoint(checkpoint, model, optimizer=None):
    """Load weights from a previous checkpoint
    Args:
        checkpoint: (dict) checkpoint as loaded in model_choice.py
        model: model to replace
        optimizer: optimiser to be used
    """
    # Corrects exception with test loop. Problem with loading generic checkpoint into DataParallel model
    # https://github.com/bearpaw/pytorch-classification/issues/27
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    if isinstance(model, nn.DataParallel) and not list(checkpoint['model'].keys())[0].startswith('module'):
        new_state_dict = model.state_dict().copy()
        new_state_dict['model'] = {'module.'+k: v for k, v in checkpoint['model'].items()}    # Very flimsy
        del checkpoint
        checkpoint = {}
        checkpoint['model'] = new_state_dict['model']

    try:
        model.load_state_dict(checkpoint['model'])
    except RuntimeError as error:
        try:
            list_errors = str(error).split('\n\t')
            mismatched_layers = []
            for error in list_errors:
                if error.startswith('size mismatch'):
                    mismatch_layer = error.split("size mismatch for ")[1].split(":")[0]    # get name of problematic layer
                    print(f'Oups. {error}. We will try chopping "{mismatch_layer}" out of pretrained dictionary.')
                    mismatched_layers.append(mismatch_layer)
            chopped_checkpt = chop_layer(checkpoint['model'], layer_names=mismatched_layers)
            # overwrite entries in the existing state dict
            model.load_state_dict(chopped_checkpt, strict=False)
        except RuntimeError as error:
            raise RuntimeError(error)

    print(f"=> loaded model")
    if optimizer and 'optimizer' in checkpoint.keys():    # 2nd condition if loading a model without optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def image_reader_as_array(input_image, scale=None, aux_vector_file=None, aux_vector_attrib=None, aux_vector_ids=None,
                          aux_vector_dist_maps=False, aux_vector_dist_log=True, aux_vector_scale=None):
    """Read an image from a file and return a 3d array (h,w,c)
    Args:
        input_image: Rasterio file handle holding the (already opened) input raster
        scale: optional scaling factor for the raw data
        aux_vector_file: optional vector file from which to extract auxiliary shapes
        aux_vector_attrib: optional vector file attribute name to parse in order to fetch ids
        aux_vector_ids: optional vector ids to target in the vector file above
        aux_vector_dist_maps: flag indicating whether aux vector bands should be distance maps or binary maps
        aux_vector_dist_log: flag indicating whether log distances should be used in distance maps or not
        aux_vector_scale: optional floating point scale factor to multiply to rasterized vector maps

    Return:
        numpy array of the image (possibly concatenated with auxiliary vector channels)
    """
    np_array = np.empty([input_image.height, input_image.width, input_image.count], dtype=np.float32)
    for i in range(input_image.count):
        np_array[:, :, i] = input_image.read(i+1)  # Bands starts at 1 in rasterio not 0

    # Guidelines for pre-processing: http://cs231n.github.io/neural-networks-2/#datapre
    # Scale arrays to values [0,1]. Default: will scale. Useful if dealing with 8 bit *and* 16 bit images.
    if scale:
        sc_min, sc_max = scale
        assert np.min(np_array) >= 0 and np.max(np_array) <= 255, f'Values in input image of shape {np_array.shape} ' \
                                                                  f'range from {np.min(np_array)} to {np.max(np_array)}.' \
                                                                  f'They should range from 0 to 255 (8bit).'
        np_array = minmax_scale(img=np_array,
                                orig_range=(0, 255),
                                scale_range=(sc_min, sc_max))

    # if requested, load vectors from external file, rasterize, and append distance maps to array
    if aux_vector_file is not None:
        vec_tensor = vector_to_raster(vector_file=aux_vector_file,
                                      input_image=input_image,
                                      attribute_name=aux_vector_attrib,
                                      fill=0,
                                      target_ids=aux_vector_ids,
                                      merge_all=False)
        if aux_vector_dist_maps:
            import cv2 as cv  # opencv becomes a project dependency only if we need to compute distance maps here
            vec_tensor = vec_tensor.astype(np.float32)
            for vec_band_idx in range(vec_tensor.shape[2]):
                mask = vec_tensor[:, :, vec_band_idx]
                mask = cv.dilate(mask, (3, 3))  # make points and linestring easier to work with
                #display_resize = cv.resize(np.where(mask, np.uint8(0), np.uint8(255)), (1000, 1000))
                #cv.imshow("mask", display_resize)
                dmap = cv.distanceTransform(np.where(mask, np.uint8(0), np.uint8(255)), cv.DIST_L2, cv.DIST_MASK_PRECISE)
                if aux_vector_dist_log:
                    dmap = np.log(dmap + 1)
                #display_resize = cv.resize(cv.normalize(dmap, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F), (1000, 1000))
                #cv.imshow("dmap1", display_resize)
                dmap_inv = cv.distanceTransform(np.where(mask, np.uint8(255), np.uint8(0)), cv.DIST_L2, cv.DIST_MASK_PRECISE)
                if aux_vector_dist_log:
                    dmap_inv = np.log(dmap_inv + 1)
                #display_resize = cv.resize(cv.normalize(dmap_inv, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F), (1000, 1000))
                #cv.imshow("dmap2", display_resize)
                vec_tensor[:, :, vec_band_idx] = np.where(mask, -dmap_inv, dmap)
                #display = cv.normalize(vec_tensor[:, :, vec_band_idx], None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
                #display_resize = cv.resize(display, (1000, 1000))
                #cv.imshow("distmap", display_resize)
                #cv.waitKey(0)
        if aux_vector_scale:
            for vec_band_idx in vec_tensor.shape[2]:
                vec_tensor[:, :, vec_band_idx] *= aux_vector_scale
        np_array = np.concatenate([np_array, vec_tensor], axis=2)
    return np_array


def vector_to_raster(vector_file, input_image, attribute_name, fill=0, target_ids=None, merge_all=True):
    """Function to rasterize vector data.
    Args:
        vector_file: Path and name of reference GeoPackage
        input_image: Rasterio file handle holding the (already opened) input raster
        attribute_name: Attribute containing the identifier for a vector (may contain slashes if recursive)
        fill: default background value to use when filling non-contiguous regions
        target_ids: list of identifiers to burn from the vector file (None = use all)
        merge_all: defines whether all vectors should be burned with their identifiers in a
            single layer or in individual layers (in the order provided by 'target_ids')

    Return:
        numpy array of the burned image
    """

    # Extract vector features to burn in the raster image
    with fiona.open(vector_file, 'r') as src:
        lst_vector = [vector for vector in src]

    # Sort feature in order to priorize the burning in the raster image (ex: vegetation before roads...)
    if attribute_name is not None:
        lst_vector.sort(key=lambda vector: get_key_recursive(attribute_name, vector))

    lst_vector_tuple = {}

    # TODO: check a vector entity is empty (e.g. if a vector['type'] in lst_vector is None.)
    for vector in lst_vector:
        id = get_key_recursive(attribute_name, vector) if attribute_name is not None else None
        if target_ids is None or id in target_ids:
            if id not in lst_vector_tuple:
                lst_vector_tuple[id] = []
            if merge_all:
                # here, we assume that the id can be cast to int!
                lst_vector_tuple[id].append((vector['geometry'], int(id) if id is not None else 0))
            else:
                # if not merging layers, just use '1' as the value for each target
                lst_vector_tuple[id].append((vector['geometry'], 1))

    if merge_all:
        return rasterio.features.rasterize([v for vecs in lst_vector_tuple.values() for v in vecs],
                                           fill=fill,
                                           out_shape=input_image.shape,
                                           transform=input_image.transform,
                                           dtype=np.int16)
    else:
        burned_rasters = [rasterio.features.rasterize(lst_vector_tuple[id],
                                                      fill=fill,
                                                      out_shape=input_image.shape,
                                                      transform=input_image.transform,
                                                      dtype=np.int16) for id in lst_vector_tuple]
        return np.stack(burned_rasters, axis=-1)


def validate_num_classes(vector_file, num_classes, attribute_name, ignore_index):    # used only in images_to_samples.py
    """Validate that the number of classes in the vector file corresponds to the expected number
    Args:
        vector_file: full file path of the vector image
        num_classes: number of classes set in config.yaml
        attribute_name: name of the value field representing the required classes in the vector image file

        value_field: name of the value field representing the required classes in the vector image file
        ignore_index: (int) target value that is ignored during training and does not contribute to the input gradient
    Return:
        None
    """

    distinct_att = set()
    with fiona.open(vector_file, 'r') as src:
        for feature in src:
            distinct_att.add(get_key_recursive(attribute_name, feature))  # Use property of set to store unique values

    detected_classes = len(distinct_att) - len([ignore_index]) if ignore_index in distinct_att else len(distinct_att)

    if detected_classes != num_classes:
        raise ValueError('The number of classes in the yaml.config {} is different than the number of classes in '
                         'the file {} {}'.format (num_classes, vector_file, str(list(distinct_att))))


def list_s3_subfolders(bucket, data_path):
    list_classes = []

    client = boto3.client('s3')
    result = client.list_objects(Bucket=bucket, Prefix=data_path+'/', Delimiter='/')
    for p in result.get('CommonPrefixes'):
        if p['Prefix'].split('/')[-2] is not data_path:
            list_classes.append(p['Prefix'].split('/')[-2])
    return list_classes


def read_csv(csv_file_name, inference=False):
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
        for row in reader:
            if inference:
                assert len(row) >= 2, 'unexpected number of columns in dataset CSV description file' \
                    ' (for inference, should have two columns, i.e. raster file path and metadata file path)'
                list_values.append({'tif': row[0], 'meta': row[1]})
            else:
                assert len(row) >= 5, 'unexpected number of columns in dataset CSV description file' \
                    ' (should have five columns; see \'read_csv\' function for more details)'
                list_values.append({'tif': row[0], 'meta': row[1], 'gpkg': row[2], 'attribute_name': row[3], 'dataset': row[4]})
    if inference:
        return list_values
    else:
        return sorted(list_values, key=lambda k: k['dataset'])


def get_device_ids(number_requested):
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
                if round(mem.used/(1024**2), 1) <  1500.0 and res.gpu < 10: # Hardcoded tolerance for memory and usage
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


def get_key_def(key, config, default=None, msg=None, delete=False):
    """Returns a value given a dictionary key, or the default value if it cannot be found."""
    if isinstance(key, list):
        if len(key) <= 1:
            if msg is not None:
                raise AssertionError(msg)
            else:
                raise AssertionError("must provide at least two valid keys to test")
        for k in key:
            if k in config:
                val = config[k]
                if delete:
                    del config[k]
                return val
        return default
    else:
        if key not in config:
            return default
        else:
            val = config[key]
            if delete:
                del config[key]
            return val


def get_key_recursive(key, config):
    """Returns a value recursively given a dictionary key that may contain multiple subkeys."""
    if not isinstance(key, list):
        key = key.split("/")  # subdict indexing split using slash
    assert key[0] in config, f"missing key '{key[0]}' in metadata dictionary"
    val = config[key[0]]
    if isinstance(val, (dict, collections.OrderedDict)):
        assert len(key) > 1, "missing keys to index metadata subdictionaries"
        return get_key_recursive(key[1:], val)
    return val


def grid_images(list_imgs_pil, list_titles):
    """Visualizes image samples"""

    height = math.ceil(len(list_imgs_pil)/4)
    width = len(list_imgs_pil) if len(list_imgs_pil) < 4 else 4
    plt.figure(figsize=(width*6, height*6))
    grid_spec = gridspec.GridSpec(height, width)

    for index, zipped in enumerate(zip(list_imgs_pil, list_titles)):
        img, title = zipped
        plt.subplot(grid_spec[index])
        plt.imshow(img)
        plt.grid(False)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()

    return plt