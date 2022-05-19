from collections import OrderedDict
from pathlib import Path
from typing import Union, Sequence, Dict, Tuple, List

import geopandas as gpd
import pyproj
import pystac
import rasterio
from pystac.extensions.eo import ItemEOExtension, Band
from omegaconf import listconfig, ListConfig
from shapely.geometry import box
from solaris.utils.core import _check_rasterio_im_load

from utils.geoutils import stack_vrts, is_stac_item
from utils.logger import get_logger
from utils.verifications import validate_by_rasterio, validate_by_geopandas, assert_crs_match

logging = get_logger(__name__)  # import logging


class SingleBandItemEO(ItemEOExtension):
    """
    Single-Band Stac Item with assets by common name.
    For info on common names, see https://github.com/stac-extensions/eo#common-band-names
    """
    def __init__(self, item: pystac.Item, bands: Sequence = None):
        super().__init__(item)
        if not is_stac_item(item):
            raise TypeError(f"Expected a valid pystac.Item object. Got {type(item)}")
        self.item = item
        self._assets_by_common_name = None

        if bands is not None and len(bands) == 0:
            logging.warning(f"At least one band should be chosen if assets need to be reached")

        # Create band inventory (all available bands)
        self.bands_all = [band for band in self.asset_by_common_name.keys()]

        # Make sure desired bands are subset of inventory
        if not set(bands).issubset(set(self.bands_all)):
            raise ValueError(f"Requested bands ({bands}) should be a subset of available bands ({self.bands_all})")

        # Filter only requested bands
        self.bands_requested = {band: self.asset_by_common_name[band] for band in bands}
        logging.debug(self.bands_all)
        logging.debug(self.bands_requested)

    @property
    def asset_by_common_name(self) -> Dict:
        """
        Get assets by common band name (only works for assets containing 1 band)
        Adapted from:
        https://github.com/sat-utils/sat-stac/blob/40e60f225ac3ed9d89b45fe564c8c5f33fdee7e8/satstac/item.py#L75
        @return:
        """
        if self._assets_by_common_name is None:
            self._assets_by_common_name = OrderedDict()
            for name, a_meta in self.item.assets.items():
                bands = []
                if 'eo:bands' in a_meta.extra_fields.keys():
                    bands = a_meta.extra_fields['eo:bands']
                if len(bands) == 1:
                    eo_band = bands[0]
                    if 'common_name' in eo_band.keys():
                        common_name = eo_band['common_name']
                        if not Band.band_range(common_name):  # Hacky but easiest way to validate common names
                            raise ValueError(f'Must be one of the accepted common names. Got "{common_name}".')
                        else:
                            self._assets_by_common_name[common_name] = {'href': a_meta.href, 'name': name}
        if not self._assets_by_common_name:
            raise ValueError(f"Common names for assets cannot be retrieved")
        return self._assets_by_common_name


class AOI(object):
    """
    Object containing all data information about a single area of interest
    based on https://github.com/stac-extensions/ml-aoi
    """

    def __init__(self, raster: Union[Path, str],
                 raster_bands_request: List = None,
                 label: Union[Path, str] = None,
                 split: str = None,
                 aoi_id: str = None,
                 collection: str = None,
                 attr_field_filter: str = None,
                 attr_values_filter: Sequence = None):
        """
        @param raster: pathlib.Path or str
            Path to source imagery
        @param label: pathlib.Path or str
            Path to ground truth file. If not provided, AOI is considered only for inference purposes
        @param split: str
            Name of destination dataset for aoi. Should be 'trn', 'tst' or 'inference'
        @param aoi_id: str
            Name or id (loosely defined) of area of interest. Used to name output folders.
            Multiple AOI instances can bear the same name.
        @param collection: str
            Name of collection containing AOI. All AOIs in the same collection should never be spatially overlapping
        @param attr_field_filter: str, optional
            Name of attribute field used to filter features. If not provided all geometries in ground truth file
            will be considered.
        @param attr_values_filter: list of ints, optional
            The list of attribute values in given attribute field used to filter features from ground truth file.
            If not provided, all values in field will be considered
        """
        raster_parsed = self.parse_input_raster(csv_raster_str=raster, raster_bands_requested=raster_bands_request)
        if isinstance(raster_parsed, Tuple):
            [validate_by_rasterio(file) for file in raster_parsed]
            raster_parsed = stack_vrts(raster_parsed)
        else:
            validate_by_rasterio(raster)

        self.raster = _check_rasterio_im_load(raster_parsed)

        if label:
            validate_by_geopandas(label)
            label_bounds = gpd.read_file(label).total_bounds
            label_bounds_box = box(*label_bounds.tolist())
            raster_bounds_box = box(*list(self.raster.bounds))
            if not label_bounds_box.intersects(raster_bounds_box):
                raise ValueError(f"Features in label file {label} do not intersect with bounds of raster file "
                                 f"{self.raster.name}")
            self.label = Path(label)
            # TODO: unit test for failed CRS match
            try:
                # TODO: check if this creates overhead. Make data validation optional?
                self.crs_match, self.epsg_raster, self.epsg_label = assert_crs_match(self.raster, self.label)
            except pyproj.exceptions.CRSError as e:
                logging.warning(f"\nError while checking CRS match between raster and label."
                                f"\n{e}")
        else:
            self.label = label
            self.crs_match = self.epsg_raster = self.epsg_label = None

        if not isinstance(split, str) and split not in ['trn', 'tst', 'inference']:
            raise ValueError(f"\nDataset split should be a string: 'trn', 'tst' or 'inference'. Got {split}.")
        elif not label and (split != 'inference' or not split):
            raise ValueError(f"\nNo ground truth provided. Dataset should be left empty or set to 'inference' only. "
                             f"\nGot {split}")
        self.split = split

        if aoi_id and not isinstance(aoi_id, str):
            raise TypeError(f'AOI name should be a string. Got {aoi_id} of type {type(aoi_id)}')
        elif not aoi_id:
            aoi_id = self.raster.stem  # Defaults to name of image without suffix
        self.aoi_id = aoi_id

        if collection and not isinstance(collection, str):
            raise TypeError(f'Collection name should be a string. Got {collection} of type {type(collection)}')
        self.aoi_id = aoi_id

        if label and attr_field_filter and not isinstance(attr_field_filter, str):
            raise TypeError(f'Attribute field name should be a string.\n'
                            f'Got {attr_field_filter} of type {type(attr_field_filter)}')
        self.attr_field_filter = attr_field_filter

        if label and attr_values_filter and not isinstance(attr_values_filter, (list, listconfig.ListConfig)):
            raise TypeError(f'Attribute values should be a list.\n'
                            f'Got {attr_values_filter} of type {type(attr_values_filter)}')
        self.attr_values_filter = attr_values_filter
        logging.debug(self)

    @classmethod
    def from_dict(cls,
                  aoi_dict,
                  bands_requested: List = None,
                  attr_field_filter: str = None,
                  attr_values_filter: list = None):
        """Instanciates an AOI object from an input-data dictionary as expected by geo-deep-learning"""
        if not isinstance(aoi_dict, dict):
            raise TypeError('Input data should be a dictionary.')
        # TODO: change dataset for split
        if not {'tif', 'gpkg', 'split'}.issubset(set(aoi_dict.keys())):
            raise ValueError(f"Input data should minimally contain the following keys: \n"
                             f"'tif', 'gpkg', 'split'.")
        if not aoi_dict['gpkg']:
            logging.warning(f"No ground truth data found for {aoi_dict['tif']}.\n"
                            f"Only imagery will be processed from now on")
        if "aoi_id" not in aoi_dict.keys() or not aoi_dict['aoi_id']:
            aoi_dict['aoi_id'] = Path(aoi_dict['tif']).stem
        aoi_dict['attribute_name'] = attr_field_filter
        new_aoi = cls(
            raster=aoi_dict['tif'],
            raster_bands_request=bands_requested,
            label=aoi_dict['gpkg'],
            split=aoi_dict['split'],
            attr_field_filter=attr_field_filter,
            attr_values_filter=attr_values_filter,
            aoi_id=aoi_dict['aoi_id']
        )
        return new_aoi

    def __str__(self):
        return (
            f"\nAOI ID: {self.aoi_id}"
            f"\n\tRaster: {self.raster.name}"
            f"\n\tLabel: {self.label}"
            f"\n\tCRS match: {self.crs_match}"
            f"\n\tSplit: {self.split}"
            f"\n\tAttribute field filter: {self.attr_field_filter}"
            f"\n\tAttribute values filter: {self.attr_values_filter}"
            )

    @staticmethod
    def parse_input_raster(csv_raster_str: str, raster_bands_requested: List) -> Union[str, Tuple]:
        # TODO: add documentation to a README somewhere
        """
        From input csv, determine if imagery is
        1. A Stac Item with single-band assets (multi-band assets not implemented)
        2. Single-band imagery as path or url with hydra-like interpolation for band identification
        3. Multi-band path or url
        @param csv_raster_str:
            input imagery to parse
        @param raster_bands_requested:
            dataset configuration parameters
        @return:
        """

        if is_stac_item(csv_raster_str):
            item = SingleBandItemEO(item=pystac.Item.from_file(csv_raster_str), bands=raster_bands_requested)
            raster = [value['href'] for value in item.bands_requested.values()]
            return tuple(raster)
        elif "${dataset.bands}" in csv_raster_str:
            if not isinstance(raster_bands_requested, (List, ListConfig)) or len(raster_bands_requested) == 0:
                raise ValueError(f"\nRequested bands should a list of bands. "
                                 f"\nGot {raster_bands_requested} of type {type(raster_bands_requested)}")
            raster = [csv_raster_str.replace("${dataset.bands}", band) for band in raster_bands_requested]
            return tuple(raster)
        else:
            try:
                validate_by_rasterio(csv_raster_str)
                return csv_raster_str
            except (FileNotFoundError, rasterio.RasterioIOError, TypeError) as e:
                logging.critical(f"Couldn't parse input raster. Got {csv_raster_str}")
                raise e
