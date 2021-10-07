from pathlib import Path
from typing import Union, List

import fiona
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import is_valid_geom
from tqdm import tqdm

from solaris_gdl.utils.core import _check_gdf_load, _check_crs, _check_rasterio_im_load
from utils.utils import subprocess_cmd
from utils.geoutils import lst_ids, get_key_recursive

import logging

logger = logging.getLogger(__name__)


def validate_num_classes(vector_file: Union[str, Path],
                         num_classes: int,
                         attribute_name: str,
                         ignore_index: int = None,
                         target_ids: List = None):
    """Check that `num_classes` is equal to number of classes detected in the specified attribute for each GeoPackage.
    Args:
        :param vector_file: full file path of the vector image
        :param num_classes: number of classes set in config_template.yaml
        :param attribute_name: name of the value field representing the required classes in the vector image file
        :param ignore_index: (int) target value that is ignored during training and does not contribute to
                             the input gradient
        :param target_ids: list of identifiers to burn from the vector file (None = use all)
    Return:
        List of unique attribute values found in gpkg vector file
    """
    if isinstance(vector_file, str):
        vector_file = Path(vector_file)
    if not vector_file.is_file():
        raise FileNotFoundError(f"Could not locate gkpg file at {vector_file}")
    unique_att_vals = set()
    with fiona.open(vector_file, 'r') as src:
        for feature in tqdm(src, leave=False, position=1, desc=f'Scanning features'):
            # Use property of set to store unique values
            unique_att_vals.add(int(get_key_recursive(attribute_name, feature)))

    # if dontcare value is defined, remove from list of unique attribute values for verification purposes
    if ignore_index and ignore_index in unique_att_vals:
        unique_att_vals.remove(ignore_index)

    # if burning a subset of gpkg's classes
    if target_ids:
        if not len(target_ids) == num_classes:
            raise ValueError(f'Yaml parameters mismatch. \n'
                             f'Got target_ids {target_ids} (sample sect) with length {len(target_ids)}. '
                             f'Expected match with num_classes {num_classes} (global sect))')
        # make sure target ids are a subset of all attribute values in gpkg
        if not set(target_ids).issubset(unique_att_vals):
            logging.warning(f'\nFailed scan of vector file: {vector_file}\n'
                            f'\tExpected to find all target ids {target_ids}. \n'
                            f'\tFound {unique_att_vals} for attribute "{attribute_name}"')
    else:
        # this can happen if gpkg doens't contain all classes, thus the warning rather than exception
        if len(unique_att_vals) < num_classes:
            logging.warning(f'Found {str(list(unique_att_vals))} classes in file {vector_file}. Expected {num_classes}')
        # this should not happen, thus the exception raised
        elif len(unique_att_vals) > num_classes:
            logging.error(f'Found {str(list(unique_att_vals))} classes in file {vector_file}. Expected {num_classes}')

    return unique_att_vals


def add_background_to_num_class(task: str, num_classes: int):
    # FIXME temporary patch for num_classes problem.
    """
    Adds one to number of classes for all segmentation tasks.

    param task: (str) task to perform. Either segmentation or classification
    param num_classes: (int) number of classes in task

    Returns number of classes corrected (+1) if task is segmentation
    """
    if task == 'segmentation':
        # assume background is implicitly needed (makes no sense to predict with one class, for example.)
        # this will trigger some warnings elsewhere, but should succeed nonetheless
        return num_classes + 1  # + 1 for background
    elif task == 'classification':
        return num_classes
    else:
        raise NotImplementedError(f'Task should be either classification or segmentation. Got "{task}"')


def assert_crs_match(raster_path: Union[str, Path], gpkg_path: Union[str, Path]):
    """
    Assert Coordinate reference system between raster and gpkg match.
    :param raster_path: (str or Path) path to raster file
    :param gpkg_path: (str or Path) path to gpkg file
    """
    raster = _check_rasterio_im_load(raster_path)
    epsg_raster = _check_crs(raster.crs.to_epsg())

    gt = _check_gdf_load(gpkg_path)
    epsg_gt = _check_crs(gt.crs.to_epsg())
    if epsg_raster != epsg_gt:
        logging.error(f"CRS mismatch: \n"
                      f"TIF file \"{raster_path}\" has {epsg_raster} CRS; \n"
                      f"GPKG file \"{gpkg_path}\" has {epsg_gt} CRS.")
        return False, epsg_raster, epsg_gt
    else:
        return True, epsg_raster, epsg_gt


def validate_features_from_gpkg(gpkg: Union[str, Path], attribute_name: str):
    """
    Validate features in gpkg file
    :param gpkg: (str or Path) path to gpkg file
    :param attribute_name: name of the value field representing the required classes in the vector image file
    """
    # TODO: test this with invalid features.
    invalid_features_list = []
    # Validate vector features to burn in the raster image
    src = _check_gdf_load(gpkg)
    for geom in tqdm(src.iterfeatures(), desc=f'Checking features'):
        if not is_valid_geom(geom['geometry']):
            invalid_features_list.append(geom['id'])
    if len(invalid_features_list) > 0:
        logging.error(f"{gpkg}: Invalid geometry object(s) '{invalid_features_list}'")
        return False, invalid_features_list
    else:
        logging.info(f"{gpkg}: Valid")
        return True, invalid_features_list


def validate_raster(raster_path: Union[str, Path], verbose: bool = True, extended: bool = False):
    """
    Checks if raster is valid, i.e. not corrupted (based on metadata, or actual byte info if under size threshold)
    @param raster_path: Path to raster to validate
    @param verbose: if True, will output potential errors detected
    @param extended: if True, rasters will be entirely read to detect any problem
    @return:
    """
    if not raster_path:
        return False, None
    raster_path = Path(raster_path) if isinstance(raster_path, str) else raster_path
    metadata = {}
    try:
        logging.info(f'Raster to validate: {raster_path}\n'
                     f'Size: {raster_path.stat().st_size}\n'
                     f'Extended check: {extended}')
        metadata = get_raster_meta(raster_path)
        if extended:
            logging.info(f'Will perform extended check\n'
                         f'Will read first band: {raster_path}')
            with rasterio.open(raster_path, 'r') as raster:
                raster_np = raster.read(1)
            logging.debug(raster_np.shape)
            if not np.any(raster_np):
                # maybe it's a valid raster filled with no data. Double check with PIL
                pil_img = Image.open(raster_path)
                pil_np = np.asarray(pil_img)
                if len(pil_np.shape) == 0 or pil_np.size <= 1:
                    logging.error(f'Corrupted raster: {raster_path}\n'
                                  f'Shape: {(pil_np.shape)}\n'
                                  f'Size: {(pil_np.size)}')
                    return False, metadata
        return True, metadata
    except rasterio.errors.RasterioIOError as e:
        if verbose:
            logging.error(e)
        return False, metadata
    except Exception as e:
        if verbose:
            logging.error(e)
        return False, metadata


def get_raster_meta(raster_path: Union[str, Path]):
    """
    Get a raster's metadata as provided by rasterio
    @param raster_path: Path to raster for which metadata is desired
    @return: (dict) Dictionary of raster's metadata (driver, dtype, nodata, width, height, count, crs, transform, etc.)
    """
    with rasterio.open(raster_path, 'r') as raster:
        metadata = raster.meta
    return metadata


def is_gdal_readable(file):
    """
    Checks if a file is a raster that can be read by GDAL
    @param file: path to file
    @return:
    """
    rcode = subprocess_cmd(f'gdalinfo {file}')
    if rcode == 0:
        return True
    else:
        return False