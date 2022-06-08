import os
from pathlib import Path
from typing import Union, List

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from fiona._err import CPLE_OpenFailedError
from fiona.errors import DriverError
from rasterio.features import is_valid_geom
from solaris.utils.core import _check_rasterio_im_load, _check_gdf_load, _check_crs
from tqdm import tqdm

from utils.geoutils import lst_ids, get_key_recursive

import logging

from utils.utils import is_url

logger = logging.getLogger(__name__)


def validate_num_classes(vector_file: Union[str, Path],
                         num_classes: int,
                         attribute_name: str,
                         ignore_index: int,
                         attribute_values: List):
    """Check that `num_classes` is equal to number of classes detected in the specified attribute for each GeoPackage.
    # FIXME: use geopandas
    FIXME: this validation **will not succeed** if a Geopackage contains only a subset of `num_classes` (e.g. 3 of 4).
    Args:
        :param vector_file: full file path of the vector image
        :param num_classes: number of classes set in old_config_template.yaml
        :param attribute_name: name of the value field representing the required classes in the vector image file
        :param ignore_index: (int) target value that is ignored during training and does not contribute to
                             the input gradient
        :param attribute_values: list of identifiers to burn from the vector file (None = use all)
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
    if ignore_index in unique_att_vals:
        unique_att_vals.remove(ignore_index)

    # if burning a subset of gpkg's classes
    if attribute_values:
        if not len(attribute_values) == num_classes:
            raise ValueError(f'Yaml parameters mismatch. \n'
                             f'Got values {attribute_values} (sample sect) with length {len(attribute_values)}. '
                             f'Expected match with num_classes {num_classes} (global sect))')
        # make sure target ids are a subset of all attribute values in gpkg
        if not set(attribute_values).issubset(unique_att_vals):
            logging.warning(f'\nFailed scan of vector file: {vector_file}\n'
                            f'\tExpected to find all target ids {attribute_values}. \n'
                            f'\tFound {unique_att_vals} for attribute "{attribute_name}"')
    else:
        # this can happen if gpkg doens't contain all classes, thus the warning rather than exception
        if len(unique_att_vals) < num_classes:
            logging.warning(f'Found {str(list(unique_att_vals))} classes in file {vector_file}. Expected {num_classes}')
        # this should not happen, thus the exception raised
        elif len(unique_att_vals) > num_classes:
            raise ValueError(
                f'Found {str(list(unique_att_vals))} classes in file {vector_file}. Expected {num_classes}')
    num_classes_ = set([i for i in range(num_classes + 1)])
    return num_classes_


def validate_raster(raster: Union[str, Path, rasterio.DatasetReader], extended: bool = False) -> None:
    """
    Checks if raster is valid, i.e. not corrupted (based on metadata, or actual byte info if under size threshold)
    @param raster: Path to raster to be validated
    @param extended: if True, raster data will be entirely read to detect any problem
    @return: if raster is valid, returns True, else False (with logging.critical)
    """
    if not raster:
        raise FileNotFoundError(f"No raster provided. Got: {raster}")
    try:
        raster = Path(raster) if isinstance(raster, str) and not is_url(raster) else raster
    except TypeError as e:
        logging.critical(f"Invalid raster.\nRaster path: {raster}\n{e}")
        raise e
    try:
        logging.debug(f'Raster to validate: {raster}\n'
                      f'Size: {raster.stat().st_size}\n'
                      f'Extended check: {extended}')
        with rasterio.open(raster, 'r') as raster:
            if not raster.meta['dtype'] in ['uint8', 'uint16']:  # will trigger exception if invalid raster
                logging.warning(f"Only uint8 and uint16 are supported in current version.\n"
                                f"Datatype {raster.meta['dtype']} for {raster.aoi_id} may cause problems.")
        if extended:
            logging.debug(f'Will perform extended check.\nWill read first band: {raster}')
            with rasterio.open(raster, 'r') as raster:
                raster_np = raster.read(1)
            logging.debug(raster_np.shape)
            if not np.any(raster_np):
                logging.critical(f"Raster data filled with zero values.\nRaster path: {raster}")
                return False
    except FileNotFoundError as e:
        logging.critical(f"Could not locate raster file.\nRaster path: {raster}\n{e}")
        raise e
    except (rasterio.errors.RasterioIOError, TypeError) as e:
        logging.critical(f"\nRasterio can't open the provided raster: {raster}\n{e}")
        raise e


def validate_num_bands(raster_path: Union[str, Path], num_bands: int) -> None:
    """
    Checks match between expected and actual number of bands
    @param raster_path: Path to raster to be validated
    @param num_bands: Number of bands expected
    @return: if expected and actual number of bands match, returns True, else False (with logging.critical)
    """
    raster = _check_rasterio_im_load(raster_path)
    input_band_count = raster.meta['count']
    if not input_band_count == num_bands:
        logging.critical(f"The number of bands expected doesn't match number of bands in input image.\n"
                         f"Expected: {num_bands} bands\n"
                         f"Got: {input_band_count} bands\n"
                         f"Raster path: {raster.name}")
        raise ValueError()


def validate_input_imagery(raster_path: Union[str, Path], num_bands: int, extended: bool = False) -> bool:
    """
    Validates raster and checks match between expected and actual number of bands
    @param raster_path: Path to raster to be validated
    @param extended: if True, raster data will be entirely read to detect any problem
    @param num_bands: Number of bands expected
    @return:
    """
    try:
        validate_raster(raster_path, extended)
    except Exception as e:  # TODO: address with issue #310
        return False
    try:
        validate_num_bands(raster_path, num_bands)
    except Exception as e:
        return False
    return True


def assert_crs_match(
        raster: Union[str, Path, rasterio.DatasetReader],
        label: Union[str, Path, gpd.GeoDataFrame]):
    """
    Assert Coordinate reference system between raster and gpkg match.
    :param raster: (str or Path) path to raster file
    :param label: (str or Path) path to gpkg file
    """
    raster = _check_rasterio_im_load(raster)
    raster_crs = raster.crs
    gt = _check_gdf_load(label)
    gt_crs = gt.crs

    epsg_gt = _check_crs(gt_crs.to_epsg())
    try:
        if raster_crs.is_epsg_code:
            epsg_raster = _check_crs(raster_crs.to_epsg())
        else:
            logging.warning(f"Cannot parse epsg code from raster's crs '{raster.name}'")
            return False, raster_crs, gt_crs

        if epsg_raster != epsg_gt:
            logging.error(f"CRS mismatch: \n"
                          f"TIF file \"{raster}\" has {epsg_raster} CRS; \n"
                          f"GPKG file \"{label}\" has {epsg_gt} CRS.")
            return False, raster_crs, gt_crs
        else:
            return True, raster_crs, gt_crs
    except AttributeError as e:
        logging.critical(f'Problem reading crs from image or label.')
        logging.critical(e)
        return False, raster_crs, gt_crs


def validate_features_from_gpkg(label: Union[str, Path], attribute_name: str):
    """
    Validate features in gpkg file
    :param label: (str or Path) path to gpkg file
    :param attribute_name: name of the value field representing the required classes in the vector image file
    """
    # FIXME: use geopandas
    # TODO: test this with invalid features.
    invalid_features_list = []
    # Validate vector features to burn in the raster image
    with fiona.open(label, 'r') as src:
        lst_vector = [vector for vector in src]
    shapes = lst_ids(list_vector=lst_vector, attr_name=attribute_name)
    for index, item in enumerate(tqdm([v for vecs in shapes.values() for v in vecs], leave=False, position=1)):
        feature_id = lst_vector[index]["id"]
        # geom must be a valid GeoJSON geometry type and non-empty
        geom, value = item
        geom = getattr(geom, '__geo_interface__', None) or geom
        if not is_valid_geom(geom):
            if feature_id not in invalid_features_list:  # ignore if feature is already appended
                if index == 0:
                    logging.critical(f"Label file contains at least one invalid feature: {label}")
                invalid_features_list.append(feature_id)
                logging.critical(f"Invalid geometry object: '{feature_id}'")
    return invalid_features_list


def validate_by_geopandas(label: Union[Path, str]):
    # TODO: unit test for valid/invalid label file
    """Check if `label` is readable by geopandas, if not, log and raise error."""
    # adapted from https://github.com/CosmiQ/solaris/blob/main/solaris/utils/core.py#L52
    if not Path(label).is_file() or os.stat(label).st_size == 0:
        raise FileNotFoundError(f'{label} is not a valid file')
    try:
        return gpd.read_file(label)
    except (DriverError, CPLE_OpenFailedError) as e:
        logging.error(
            f"GeoDataFrame couldn't be loaded: either {label} isn't a valid"
            " path or it isn't a valid vector file. Returning an empty"
            " GeoDataFrame."
        )
        raise e
