import logging

import numpy as np
import rasterio

from utils.geoutils import clip_raster_with_gpkg

logger = logging.getLogger(__name__)


def image_reader_as_array(input_image,
                          clip_gpkg=None,
                          debug=False):
    """Read an image from a file and return a 3d array (h,w,c)
    Args:
        input_image: Rasterio file handle holding the (already opened) input raster
        clip_gpkg: optional path to gpkg used to clip input_image
        debug: if True, output raster as given by clip_raster_with_gpkg function is saved to disk

    Return:
        numpy array of the image
    """
    if clip_gpkg:
        try:
            clipped_img_pth = clip_raster_with_gpkg(input_image, clip_gpkg, debug=debug)
            input_image = rasterio.open(clipped_img_pth, 'r')
            np_array = input_image.read()
        except ValueError:  # if gpkg's extent outside raster: "ValueError: Input shapes do not overlap raster."
            logging.exception(f'Problem clipping raster with geopackage {clip_gpkg}')
            np_array = input_image.read()
    else:
        np_array = input_image.read()

    np_array = np.moveaxis(np_array, 0, -1)  # send channels last
    if np_array.dtype not in ['uint8', 'uint16']:
        raise NotImplementedError(f"Invalid datatype {np_array.dtype}. "
                                  f"Only uint8 and uint16 are supported in current version")

    dataset_nodata = None
    if input_image.nodata is not None:
        # See: https://rasterio.readthedocs.io/en/latest/topics/masks.html#dataset-masks
        dataset_nodata = np_array == input_image.nodata
        # Keep only nodata when present across all bands
        dataset_nodata = np.all(dataset_nodata, axis=2)

    return np_array, input_image, dataset_nodata
