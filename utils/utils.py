import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import os
from torch import nn
import numpy as np
import rasterio
import warnings
import fiona
import csv

from preprocess import minmax_scale

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


def create_or_empty_folder(folder):
    """Empty an existing folder or create it if it doesn't exist.
    Args:
        folder: full file path of the folder to be emptied/created
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
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


def chop_layer(pretrained_dict, layer_names=["logits"]):   #https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    """
    Removes keys from a layer in state dictionary of model architecture.
    :param model: (nn.Module) model with original architecture
    :param layer_names: (list) names of layers to be chopped.
    :return: (nn.Module) model
    """
    # filter out weights from undesired keys. ex.: size mismatch.
    for layer in layer_names:
        chopped_dict = {k: v for k, v in pretrained_dict.items() if k.find(layer) == -1}
        pretrained_dict = chopped_dict    # overwrite values in pretrained dict with chopped dict
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


def image_reader_as_array(input_image, scale=None, aux_vector_file=None, aux_vector_attrib=None,
                          aux_vector_ids=None, aux_vector_dist_maps=False, aux_vector_scale=None):
    """Read an image from a file and return a 3d array (h,w,c)
    Args:
        input_image: Rasterio file handle holding the (already opened) input raster
        scale: optional scaling factor for the raw data
        aux_vector_file: optional vector file from which to extract auxiliary shapes
        aux_vector_attrib: optional vector file attribute name to parse in order to fetch ids
        aux_vector_ids: optional vector ids to target in the vector file above
        aux_vector_dist_maps: flag indicating whether aux vector bands should be distance maps or binary maps
        aux_vector_scale: optional floating point scale factor to multiply to rasterized vector maps

    Return:
        numpy array of the image (possibly concatenated with auxiliary vector channels)
    """
    np_array = np.empty([input_image.height, input_image.width, input_image.count], dtype=np.float32)
    for i in range(input_image.count):
        band = input_image.read(i+1)  # Bands starts at 1 in rasterio not 0
        np_array[:, :, i] = band

    # Guidelines for pre-processing: http://cs231n.github.io/neural-networks-2/#datapre
    # Scale arrays to values [0,1]. Default: will scale. Useful if dealing with 8 bit *and* 16 bit images.
    if scale:
        sc_min, sc_max = scale
        np_array = minmax_scale(img=np_array,
                                orig_range=(np.min(np_array), np.max(np_array)),
                                scale_range=(sc_min, sc_max))

    # if requested, load vectors from external file, rasterize, and append distance maps to array
    if aux_vector_file is not None:
        assert aux_vector_attrib is not None, \
            "vector file identifier attribute name must not be none; it will be used to extract target ids"
        assert aux_vector_ids is not None and aux_vector_ids, \
            "list of target vector ids must not be none; it is used to determine final tensor depth"
        vec_tensor = vector_to_raster(vector_file=aux_vector_file,
                                      input_image=input_image,
                                      attribute_name=aux_vector_attrib,
                                      fill=0,
                                      target_ids=aux_vector_ids,
                                      merge_all=False)
        if aux_vector_dist_maps:
            import cv2 as cv  # opencv becomes a project dependency only if we need to compute distance maps here
            for vec_band_idx in vec_tensor.shape[2]:
                mask = vec_tensor[:, :, vec_band_idx]
                dmap = cv.distanceTransform(np.where(mask, np.uint8(0), np.uint8(255)), cv.DIST_L2, cv.DIST_MASK_PRECISE)
                dmap_inv = cv.distanceTransform(np.where(mask, np.uint8(255), np.uint8(0)), cv.DIST_L2, cv.DIST_MASK_PRECISE)
                vec_tensor[:, :, vec_band_idx] = np.where(mask, -dmap_inv, dmap)
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
        attribute_name: Attribute containing the identifier for a vector
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
    lst_vector.sort(key=lambda vector: vector['properties'][attribute_name])

    assert merge_all or target_ids is not None, \
        "if not merging all vectors in the same layer, target id list must be provided"

    lst_vector_tuple = [] if merge_all else {tgt: [] for tgt in target_ids}

    # TODO: check a vector entity is empty (e.g. if a vector['type'] in lst_vector is None.)
    for vector in lst_vector:
        if target_ids is None or vector['properties'][attribute_name] in target_ids:
            if merge_all:
                lst_vector_tuple.append((vector['geometry'], int(vector['properties'][attribute_name])))
            else:
                lst_vector_tuple[vector['properties'][attribute_name]].append((vector['geometry'], 1))

    if merge_all:
        return rasterio.features.rasterize(lst_vector_tuple,
                                           fill=fill,
                                           out_shape=input_image.shape,
                                           transform=input_image.transform,
                                           dtype=np.uint8)
    else:
        burned_rasters = [rasterio.features.rasterize(lst_vector_tuple[tgt],
                                                      fill=fill,
                                                      out_shape=input_image.shape,
                                                      transform=input_image.transform,
                                                      dtype=np.uint8) for tgt in target_ids]
        return np.stack(burned_rasters, axis=-1)


def validate_num_classes(vector_file, num_classes, value_field):    # used only in images_to_samples.py
    """Validate that the number of classes in the vector file corresponds to the expected number
    Args:
        vector_file: full file path of the vector image
        num_classes: number of classes set in config.yaml
        value_field: name of the value field representing the required classes in the vector image file

    Return:
        None
    """

    distinct_att = set()
    with fiona.open(vector_file, 'r') as src:
        for feature in src:
            distinct_att.add(feature['properties'][value_field])  # Use property of set to store unique values

    if len(distinct_att) != num_classes:
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
                assert len(row) == 2, 'unexpected number of columns in dataset CSV description file' \
                    ' (for inference, should have two columns, i.e. raster file path and metadata file path)'
                list_values.append({'tif': row[0]})
            else:
                assert len(row) == 5, 'unexpected number of columns in dataset CSV description file' \
                    ' (should have five columns; see \'read_csv\' function for more details)'
                list_values.append({'tif': row[0], 'meta': row[1], 'gpkg': row[2], 'attribute_name': row[3], 'dataset': row[4]})
    if inference:
        return list_values
    else:
        return sorted(list_values, key=lambda k: k['dataset'])


def get_device_ids(number_requested): #FIXME if some memory is used on a GPU before call to this function, the GPU will be excluded.
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
