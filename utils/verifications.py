from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import rasterio

import logging

from utils.geoutils import check_rasterio_im_load, check_gdf_load, check_crs
from utils.utils import is_url

logger = logging.getLogger(__name__)


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
        raster = check_rasterio_im_load(raster)
    except (TypeError, ValueError) as e:
        logging.critical(f"Invalid raster.\nRaster path: {raster}\n{e}")
        raise e
    try:
        size = Path(raster.name).stat().st_size if not is_url(raster.name) else None
        logging.debug(f'Raster to validate: {raster}\n'
                      f'Size: {size}\n'
                      f'Extended check: {extended}')
        if not raster.meta['dtype'] in ['uint8', 'uint16']:  # will trigger exception if invalid raster
            logging.warning(f"Only uint8 and uint16 are supported in current version.\n"
                            f"Datatype {raster.meta['dtype']} for {raster.aoi_id} may cause problems.")
        if extended:
            logging.debug(f'Will perform extended check.\nWill read first band: {raster}')
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
    raster = check_rasterio_im_load(raster_path)
    input_band_count = raster.meta['count']
    if not input_band_count == num_bands:
        logging.critical(f"The number of bands expected doesn't match number of bands in input image.\n"
                         f"Expected: {num_bands} bands\n"
                         f"Got: {input_band_count} bands\n"
                         f"Raster path: {raster.name}")
        raise ValueError()


def assert_crs_match(
        raster: Union[str, Path, rasterio.DatasetReader],
        label: Union[str, Path, gpd.GeoDataFrame]):
    """
    Assert Coordinate reference system between raster and gpkg match.
    :param raster: (str or Path) path to raster file
    :param label: (str or Path) path to gpkg file
    """
    raster = check_rasterio_im_load(raster)
    raster_crs = raster.crs
    gt = check_gdf_load(label)
    gt_crs = gt.crs

    epsg_gt = check_crs(gt_crs.to_epsg())
    try:
        if raster_crs.is_epsg_code:
            epsg_raster = check_crs(raster_crs.to_epsg())
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


def validate_features_from_gpkg(label: Union[str, Path, gpd.GeoDataFrame]):
    """
    Validate features in gpkg file
    :param label: (str or Path) path to gpkg file
    """
    label_gdf = check_gdf_load(label)
    invalid_features = list(np.where(label_gdf.is_valid != True)[0])
    return invalid_features
