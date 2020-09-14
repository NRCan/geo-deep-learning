import numpy as np
from ruamel_yaml import YAML
from tqdm import tqdm
from pathlib import Path
from skimage import morphology

from utils.geoutils import vector_to_raster, clip_raster_with_gpkg


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


def image_reader_as_array(input_image,
                          aux_vector_file=None,
                          aux_vector_attrib=None,
                          aux_vector_ids=None,
                          aux_vector_dist_maps=False,
                          aux_vector_dist_log=True,
                          aux_vector_scale=None,
                          clip_gpkg=None,
                          debug=False):
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
        clip_gpkg: optional path to gpkg used to clip input_image
        debug: if True, output raster as given by clip_raster_with_gpkg function is saved to disk

    Return:
        numpy array of the image (possibly concatenated with auxiliary vector channels)
    """
    if clip_gpkg:
        np_array, input_image = clip_raster_with_gpkg(input_image, clip_gpkg, debug=debug)
    else:
        np_array = input_image.read()

    np_array = np.moveaxis(np_array, 0, -1) # send channels last
    assert np_array.dtype in ['uint8', 'uint16'], f"Invalid datatype {np_array.dtype}. " \
                                                  f"Only uint8 and uint16 are supported in current version"

    dataset_nodata = None
    if input_image.nodata is not None:
        # See: https://rasterio.readthedocs.io/en/latest/topics/masks.html#dataset-masks
        dataset_nodata = np_array == input_image.nodata
        # Keep only nodata when present across all bands
        dataset_nodata = np.all(dataset_nodata, axis=2)

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
                kernel = np.ones(3, 3)
                # mask = cv.dilate(mask, kernel)  # make points and linestring easier to work with
                mask = morphology.binary_dilation(mask, kernel)  # make points and linestring easier to work with
                # display_resize = cv.resize(np.where(mask, np.uint8(0), np.uint8(255)), (1000, 1000))
                # cv.imshow("mask", display_resize)
                dmap = cv.distanceTransform(np.where(mask, np.uint8(0), np.uint8(255)), cv.DIST_L2,
                                            cv.DIST_MASK_PRECISE)
                if aux_vector_dist_log:
                    dmap = np.log(dmap + 1)
                # display_resize = cv.resize(cv.normalize(dmap, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F), (1000, 1000))
                # cv.imshow("dmap1", display_resize)
                dmap_inv = cv.distanceTransform(np.where(mask, np.uint8(255), np.uint8(0)), cv.DIST_L2,
                                                cv.DIST_MASK_PRECISE)
                if aux_vector_dist_log:
                    dmap_inv = np.log(dmap_inv + 1)
                # display_resize = cv.resize(cv.normalize(dmap_inv, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F), (1000, 1000))
                # cv.imshow("dmap2", display_resize)
                vec_tensor[:, :, vec_band_idx] = np.where(mask, -dmap_inv, dmap)
                # display = cv.normalize(vec_tensor[:, :, vec_band_idx], None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
                # display_resize = cv.resize(display, (1000, 1000))
                # cv.imshow("distmap", display_resize)
                # cv.waitKey(0)
        if aux_vector_scale:
            for vec_band_idx in vec_tensor.shape[2]:
                vec_tensor[:, :, vec_band_idx] *= aux_vector_scale
        np_array = np.concatenate([np_array, vec_tensor], axis=2)

    return np_array, input_image, dataset_nodata


