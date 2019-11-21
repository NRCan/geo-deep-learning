import csv

import numpy as np
from ruamel_yaml import YAML
from tqdm import tqdm

from utils.utils import vector_to_raster, minmax_scale


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
    for i in tqdm(range(input_image.count), position=1, leave=False, desc=f'Reading images'):
        np_array[:, :, i] = input_image.read(i+1)  # Bands starts at 1 in rasterio not 0

    # Guidelines for pre-processing: http://cs231n.github.io/neural-networks-2/#datapre
    # Scale array values from range [0,255] to values in config (e.g. [0,1])
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