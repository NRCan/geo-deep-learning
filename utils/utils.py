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


def load_from_checkpoint(checkpoint, model, optimizer=None, inference=False):
    """Load weights from a previous checkpoint
    Args:
        checkpoint: (dict) checkpoint as loaded in model_choice.py
        model: model to replace
        optimizer: optimiser to be used
    """
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"=> loaded model\n")
    if optimizer and 'optimizer' in checkpoint.keys():    # 2nd condition if loading a model without optimizer
        optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
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
        np_array = minmax_scale(img=np_array,
                                orig_range=(np.min(np_array), np.max(np_array)),
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
        num_file = 0
        filename = '/export/sata01/wspace/dataset_kingston_rgb/tst_CRIM/Images/' + str(num_file) + '.png'
        cv.imwrite(filename, vec_tensor)
        num_file += 1
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


def lst_ids(list_vector, attr_name, target_ids=None, merge_all=True):
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


def create_new_raster_from_base(input_raster, output_raster, write_array):
    """Function to use info from input raster to create new one.
    Args:
        input_raster: input raster path and name
        output_raster: raster name and path to be created with info from input
        write_array (optional): array to write into the new raster

    Return:
        none
    """
    if len(write_array.shape) == 2:  # 2D array
        count = 1
    elif len(write_array.shape) == 3:  # 3D array
        count = 3
    else:
        raise ValueError(f'Array with {len(write_array.shape)} dimensions cannot be written by rasterio.')

    with rasterio.open(input_raster, 'r') as src:
        with rasterio.open(output_raster, 'w',
                           driver=src.driver,
                           width=src.width,
                           height=src.height,
                           count=count,
                           crs=src.crs,
                           dtype=np.uint8,
                           transform=src.transform) as dst:
            if count == 1:
                dst.write(write_array[:, :], 1)
            elif count == 3:
                dst.write(write_array[:, :, :3])  # Take only first three bands assuming they are RGB.


def unnormalize(input_img, mean, std):

    """
    :param input_img: (numpy array) Image to be "unnormalized"
    :param mean: (list of mean values) for each channel
    :param std:  (list of std values) for each channel
    :return: (numpy_array) "Unnormalized" image
    """
    return (input_img * std) + mean
