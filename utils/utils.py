import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
from torch import nn
import numpy as np
import rasterio
import rasterio.features
import warnings
import collections
import fiona
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


def load_from_checkpoint(checkpoint, model, optimizer=None): # FIXME: add boolean paramter for inference.
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
                    warnings.warn(f'Oups. {error}. We will try chopping "{mismatch_layer}" out of pretrained dictionary.')
                    mismatched_layers.append(mismatch_layer)
            chopped_checkpt = chop_layer(checkpoint['model'], layer_names=mismatched_layers)
            # overwrite entries in the existing state dict
            model.load_state_dict(chopped_checkpt, strict=False)
        except RuntimeError as error:
            raise RuntimeError(error)

    print(f"=> loaded model\n\n")
    if optimizer and 'optimizer' in checkpoint.keys():    # 2nd condition if loading a model without optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


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

    lst_vector_tuple = lst_ids(list_vector=lst_vector, attr_name=attribute_name, target_ids=target_ids, merge_all=merge_all)

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


def get_key_def(key, config, default=None, msg=None, delete=False):
    """Returns a value given a dictionary key, or the default value if it cannot be found.
    :param key: key in dictionary (e.g. generated from .yaml)
    :param config: (dict) dictionary containing keys corresponding to parameters used in script
    :param default: default value assigned if no value found with provided key
    :param msg: message returned with AssertionError si length of key is smaller or equal to 1
    :param delete: (bool) if True, deletes parameter FIXME: check if this is true. Not sure I understand why we would want to delete a parameter.
    :return:
    """
    if isinstance(key, list): # is key a list?
        if len(key) <= 1: # is list of length 1 or shorter? else --> default
            if msg is not None:
                raise AssertionError(msg)
            else:
                raise AssertionError("Must provide at least two valid keys to test")
        for k in key: # iterate through items in list
            if k in config: # if item is a key in config, set value.
                val = config[k]
                if delete: # optionally delete parameter after defining a variable with it
                    del config[k]
                return val
        return default
    else: # if key is not a list
        if key not in config or config[key] is None: # if key not in config dict
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
    assert key[0] in config, f"missing key '{key[0]}' in metadata dictionary: {config}"
    val = config[key[0]]
    if isinstance(val, (dict, collections.OrderedDict)):
        assert len(key) > 1, "missing keys to index metadata subdictionaries"
        return get_key_recursive(key[1:], val)
    return val


def lst_ids(list_vector, attr_name, target_ids=None, merge_all=True): # FIXME: documentation!
    '''
    Generates a dictionary from a list of vectors where keys are class numbers and values are corresponding features in a list.
    :param list_vector: list of vectors as returned by fiona.open
    :param attr_name: Attribute containing the identifier for a vector (may contain slashes if recursive)
    :param target_ids: list of identifiers to burn from the vector file (None = use all)
    :param merge_all: defines whether all vectors should be burned with their identifiers in a
            single layer or in individual layers (in the order provided by 'target_ids')
    :return: list of tuples in format (vector, class_id).
    '''
    lst_vector_tuple = {}
    for vector in list_vector:
        id = get_key_recursive(attr_name, vector) if attr_name is not None else None
        if target_ids is None or id in target_ids:
            if id not in lst_vector_tuple:
                lst_vector_tuple[id] = []
            if merge_all:
                # here, we assume that the id can be cast to int!
                lst_vector_tuple[id].append((vector['geometry'], int(id) if id is not None else 0))
            else:
                # if not merging layers, just use '1' as the value for each target
                lst_vector_tuple[id].append((vector['geometry'], 1))
    return lst_vector_tuple


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


