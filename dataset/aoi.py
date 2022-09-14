import functools
import json
from collections import OrderedDict
from multiprocessing import Value
from pathlib import Path
from typing import Union, Sequence, Dict, Tuple, List, Optional

import geopandas as gpd
import numpy as np
import pyproj
import pystac
import rasterio
from hydra.utils import to_absolute_path
from pandas.io.common import is_url
from pystac.extensions.eo import ItemEOExtension, Band
from omegaconf import listconfig, ListConfig
from shapely.geometry import box, Polygon, MultiPolygon
from solaris.utils.core import _check_rasterio_im_load, _check_gdf_load
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from utils.geoutils import stack_singlebands_vrt, is_stac_item, create_new_raster_from_base
from utils.logger import get_logger
from utils.utils import read_csv
from utils.verifications import assert_crs_match, validate_raster, \
    validate_num_bands, validate_features_from_gpkg

logging = get_logger(__name__)  # import logging


class SingleBandItemEO(ItemEOExtension):
    """
    Single-Band Stac Item with assets by common name.
    For info on common names, see https://github.com/stac-extensions/eo#common-band-names
    """
    def __init__(
            self,
            item: pystac.Item,
            bands_requested: Optional[Sequence] = None,
    ):
        """

        @param item:
            Stac item containing metadata linking imagery assets
        @param bands_requested:
            band selection which must be a list of STAC Item common names from eo extension.
            See: https://github.com/stac-extensions/eo/#common-band-names
        """
        super().__init__(item)
        if not is_stac_item(item):
            raise TypeError(f"Expected a valid pystac.Item object. Got {type(item)}")
        self.item = item
        self._assets_by_common_name = None

        if bands_requested is not None and len(bands_requested) == 0:
            logging.warning(f"At least one band should be chosen if assets need to be reached")

        # Create band inventory (all available bands)
        self.bands_all = [band for band in self.asset_by_common_name.keys()]

        # Make sure desired bands are subset of inventory
        if not set(bands_requested).issubset(set(self.bands_all)):
            raise ValueError(f"Requested bands ({bands_requested}) should be a subset of available bands ({self.bands_all})")

        # Filter only requested bands
        self.bands_requested = {band: self.asset_by_common_name[band] for band in bands_requested}
        logging.debug(self.bands_all)
        logging.debug(self.bands_requested)

        bands = []
        for band in self.bands_requested.keys():
            band = Band.create(
                name=self.bands_requested[band]['name'],
                common_name=band,
                description=self.bands_requested[band]['meta'].description,
                center_wavelength=self.bands_requested[band]['meta'].extra_fields['eo:bands'][0]['center_wavelength'],
                full_width_half_max=self.bands_requested[band]['meta'].extra_fields['eo:bands'][0]['full_width_half_max'])
            bands.append(band)
        self.bands = bands

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
                            self._assets_by_common_name[common_name] = {'meta': a_meta, 'name': name}
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
                 raster_num_bands_expected: int = None,
                 attr_field_filter: str = None,
                 attr_values_filter: Sequence = None,
                 download_data: bool = False,
                 root_dir: str = "data",
                 for_multiprocessing: bool = False,
                 raster_stats: bool = False,
                 write_multiband: bool = False):
        # TODO: dict printer to output report on list of aois
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
        @param raster_num_bands_expected:
            Number of bands expected in processed raster (e.g. after combining single-bands files into a VRT)
        @param attr_field_filter: str, optional
            Name of attribute field used to filter features. If not provided all geometries in ground truth file
            will be considered.
        @param attr_values_filter: list of ints, optional
            The list of attribute values in given attribute field used to filter features from ground truth file.
            If not provided, all values in field will be considered
        @param download_data:
            if True, download dataset and store it in the root directory.
        @param root_dir:
            root directory where dataset can be found or downloaded
        @param raster_stats:
            if True, radiometric stats will be read from Stac Item if available or calculated
        @param write_multiband: bool, optional
            If True, a multi-band raster side by side with single-bands rasters as provided in input csv. For debugging purposes.
        @param for_multiprocessing: bool, optional
            If True, no rasterio.DatasetReader will be generated in __init__. User will have to call read raster later.
            See: https://github.com/rasterio/rasterio/issues/1731
        """
        self.raster_multiband = None
        self.raster_np = None
        self.raster_closed = False

        # Check and parse raster data
        if not isinstance(raster, str):
            raise TypeError(f"Raster path should be a string.\nGot {raster} of type {type(raster)}")
        self.raster_raw_input = raster
        if raster_bands_request and not isinstance(raster_bands_request, (Sequence, ListConfig)):
            raise ValueError(f"Requested bands should be a list."
                             f"\nGot {raster_bands_request} of type {type(raster_bands_request)}")
        self.raster_bands_request = raster_bands_request
        raster_parsed = self.parse_input_raster(
            csv_raster_str=self.raster_raw_input,
            raster_bands_requested=self.raster_bands_request
        )

        # If stac item input, keep Stac item object as attribute
        if is_stac_item(self.raster_raw_input):
            item = SingleBandItemEO(item=pystac.Item.from_file(self.raster_raw_input),
                                    bands_requested=self.raster_bands_request)
            self.raster_stac_item = item
        else:
            self.raster_stac_item = None

        # If parsed result has more than a single file, then we're dealing with single-band files
        self.raster_src_is_multiband = True if len(raster_parsed) == 1 else False

        # Download assets if desired
        self.download_data = download_data
        self.root_dir = Path(root_dir)

        if self.download_data:
            for index, single_raster in enumerate(raster_parsed):
                if is_url(single_raster):
                    out_name = self.root_dir / Path(single_raster).name
                    download_url(single_raster, root=str(out_name.parent), filename=out_name.name)
                    # replace with local copy
                    raster_parsed[index] = str(out_name)

        # validate raster data
        for single_raster in raster_parsed:
            validate_raster(single_raster)
        self.raster_parsed = raster_parsed

        # if single band assets, build multiband VRT
        self.raster_to_multiband(virtual=True)
        self.raster_read()
        self.raster_meta = self.raster.meta
        self.raster_meta['name'] = self.raster.name
        if self.raster_src_is_multiband:
            self.raster_name = Path(self.raster.name)
        else:
            self.raster_name = Path(self.raster_raw_input[0]).name.replace("${dataset.bands}", "")

        if raster_num_bands_expected:
            validate_num_bands(raster_path=self.raster, num_bands=raster_num_bands_expected)

        if self.raster_parsed and write_multiband:
            self.write_multiband_from_singleband_rasters_as_vrt()

        # Check label data
        if label:
            self.label = Path(label)
            self.label_gdf = _check_gdf_load(str(label))
            self.bounds_iou = self.bounds_iou_gdf_riodataset(
                gdf=self.label_gdf,
                raster=self.raster)
            if self.bounds_iou == 0:
                logging.error(
                    f"Features in label file {label} do not intersect with bounds of raster file "
                    f"{self.raster.name}")
            self.label_invalid_features = validate_features_from_gpkg(label)

            # TODO: unit test for failed CRS match
            try:
                # TODO: check if this creates overhead. Skip if report exists?
                self.crs_match, self.epsg_raster, self.epsg_label = assert_crs_match(self.raster, self.label_gdf)
            except pyproj.exceptions.CRSError as e:
                logging.warning(f"\nError while checking CRS match between raster and label."
                                f"\n{e}")
        else:
            self.label = self.label_gdf = self.crs_match = self.epsg_raster = self.epsg_label = None

        # Check split string
        if split and not isinstance(split, str):
            raise ValueError(f"\nDataset split should be a string.\nGot {split}.")

        if label and split not in ['trn', 'tst', 'inference']:
            raise ValueError(f"\nWith ground truth, split should be 'trn', 'tst' or 'inference'. \nGot {split}")
        # force inference split if no label provided
        elif not label and (split != 'inference' or not split):
            logging.warning(f"\nNo ground truth provided. Dataset split will be set to 'inference'"
                            f"\nOriginal split: {split}")
            split = 'inference'
        self.split = split

        # Check aoi_id string
        if aoi_id and not isinstance(aoi_id, str):
            raise TypeError(f'AOI name should be a string. Got {aoi_id} of type {type(aoi_id)}')
        elif not aoi_id and self.raster_src_is_multiband:
            aoi_id = Path(self.raster.name).stem  # Defaults to name of image without suffix
        elif not aoi_id and not self.raster_src_is_multiband:
            aoi_id = Path(self.raster_raw_input).stem  # Defaults to name of first singleband image without suffix
        self.aoi_id = aoi_id

        # Check collection string
        if collection and not isinstance(collection, str):
            raise TypeError(f'Collection name should be a string. Got {collection} of type {type(collection)}')
        self.aoi_id = aoi_id

        # If ground truth is provided, check attribute field
        if label and attr_field_filter:
            if not isinstance(attr_field_filter, str):
                raise TypeError(f'Attribute field name should be a string.\n'
                                f'Got {attr_field_filter} of type {type(attr_field_filter)}')
            elif attr_field_filter not in self.label_gdf.columns:
                # fiona and geopandas don't expect attribute name exactly the same way: "properties/class" vs "class"
                attr_field_filter = attr_field_filter.split('/')[-1]
                if attr_field_filter not in self.label_gdf.columns:
                    raise ValueError(f"\nAttribute field \"{attr_field_filter}\" not found in label attributes:\n"
                                     f"{self.label_gdf.columns}")
        self.attr_field_filter = attr_field_filter

        # If ground truth is provided, check attribute values to filter from
        if isinstance(attr_values_filter, int):
            attr_values_filter = [attr_values_filter]
        if label and attr_values_filter and not isinstance(attr_values_filter, (list, listconfig.ListConfig)):
            raise TypeError(f'Attribute values should be a list.\n'
                            f'Got {attr_values_filter} of type {type(attr_values_filter)}')
        self.attr_values_filter = attr_values_filter
        if label:
            self.label_gdf_filtered = self.filter_gdf_by_attribute(
                self.label_gdf.copy(deep=True),
                self.attr_field_filter,
                self.attr_values_filter,
            )
            if len(self.label_gdf_filtered) == 0:
                logging.warning(f"\nNo features found for ground truth \"{self.label}\","
                                 f"\nfiltered by attribute field \"{self.attr_field_filter}\""
                                 f"\nwith values \"{self.attr_values_filter}\"")
        else:
            self.label_gdf_filtered = None

        self.raster_stats = self.calc_raster_stats() if raster_stats else None

        if not isinstance(for_multiprocessing, bool):
            raise ValueError(f"\n\"for_multiprocessing\" should be a boolean.\nGot {for_multiprocessing}.")
        self.for_multiprocessing = for_multiprocessing
        if self.for_multiprocessing:
            self.close_raster()
            self.raster = None

        logging.debug(self)

    @classmethod
    def from_dict(cls,
                  aoi_dict,
                  bands_requested: List = None,
                  attr_field_filter: str = None,
                  attr_values_filter: list = None,
                  download_data: bool = False,
                  root_dir: str = "data",
                  for_multiprocessing: bool = False):
        """Instanciates an AOI object from an input-data dictionary as expected by geo-deep-learning"""
        if not isinstance(aoi_dict, dict):
            raise TypeError('Input data should be a dictionary.')
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
            aoi_id=aoi_dict['aoi_id'],
            download_data=download_data,
            root_dir=root_dir,
            for_multiprocessing=for_multiprocessing,
        )
        return new_aoi

    # TODO: is this necessary if to_dict() is good enough?
    def __str__(self):
        return (
            f"\nAOI ID: {self.aoi_id}"
            f"\n\tRaster: {self.raster_name}"
            f"\n\tLabel: {self.label}"
            f"\n\tCRS match: {self.crs_match}"
            f"\n\tSplit: {self.split}"
            f"\n\tAttribute field filter: {self.attr_field_filter}"
            f"\n\tAttribute values filter: {self.attr_values_filter}"
            )

    def raster_to_multiband(self, virtual=True):
        if not self.raster_src_is_multiband:
            if virtual:
                self.raster_multiband = stack_singlebands_vrt(self.raster_parsed)
            else:
                self.raster_multiband = self.write_multiband_from_singleband_rasters_as_vrt()
        else:
            self.raster_multiband = self.raster_parsed[0]

    def raster_read(self):
        self.raster = _check_rasterio_im_load(self.raster_multiband)

    def to_dict(self, extended=True):
        """returns a dictionary containing all important attributes of AOI (ex.: to print a report or output csv)"""
        try:
            raster_area = (self.raster.res[0] * self.raster.width) * (self.raster.res[1] * self.raster.height)
        except AttributeError:
            raster_area = None
        out_dict = {
            'raster': self.raster_raw_input,
            'label': self.label,
            'split': self.split,
            'id': self.aoi_id,
            'raster_parsed': self.raster_parsed,
            'raster_area': raster_area,
            'raster_meta': self.raster_meta,
            'label_features_nb': len(self.label_gdf),
            'label_features_filtered_nb': len(self.label_gdf_filtered),
            'raster_label_bounds_iou': self.bounds_iou,
            'crs_raster': self.epsg_raster,
            'crs_label': self.epsg_label,
            'crs_match': self.crs_match
        }
        if extended:
            mean_ext_vert_nb = None
            try:
                if isinstance(list(self.label_gdf_filtered.geometry)[0], MultiPolygon):
                    ext_vert = []
                    for multipolygon in list(self.label_gdf_filtered.geometry):
                        ext_vert.extend([len(geom.exterior.coords) for geom in list(multipolygon)])
                    mean_ext_vert_nb = np.mean(ext_vert)
                elif isinstance(list(self.label_gdf_filtered.geometry)[0], Polygon):
                    mean_ext_vert_nb = np.mean([len(geom.exterior.coords) for geom in self.label_gdf_filtered.geometry])
            # TODO: resolve with MB18 ('Polygon' object is not iterable)
            # TODO: Kingston1 ('MultiPolygon' object has no attribute 'exterior')
            # TODO: AB11 (0 filtered features)
            except Exception as e:
                logging.warning(e)

            out_dict.update({
                'label_features_filtered_mean_area': np.mean(self.label_gdf_filtered.area),
                'label_features_filtered_mean_perimeter': np.mean(self.label_gdf_filtered.length),
                'label_features_filtered_mean_exterior_vertices_nb': mean_ext_vert_nb
            })
        return out_dict

    def calc_raster_stats(self):
        """ For stac items formatted as expected, reads mean and std of raster imagery, per band.
        See stac item example: tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03.json
        If source imagery is not a stac item or stac item lacks stats assets, stats are calculed on the fly"""
        if self.raster_bands_request:
            stats = {name: {} for name in self.raster_bands_request}
        else:
            stats = {f"band_{index}": {} for index in range(self.raster.count)}
        try:
            stats_asset = self.raster_stac_item.item.assets['STATS']
            if is_url(stats_asset.href):
                download_url(stats_asset.href, root=str(self.root_dir), filename=Path(stats_asset.href).name)
                stats_href = self.root_dir / Path(stats_asset.href).name
            else:
                stats_href = to_absolute_path(stats_asset.href)
            with open(stats_href, 'r') as ifile:
                stac_stats = json.loads(ifile.read())
            stac_stats = {bandwise_stats['asset']: bandwise_stats for bandwise_stats in stac_stats}
            for band in self.raster_stac_item.bands:
                stats[band.common_name] = stac_stats[band.name]
        except (AttributeError, KeyError):
            self.raster_np = self.raster.read()
            for index, band in enumerate(stats.keys()):
                stats[band] = {"statistics": {}, "histogram": {}}
                stats[band]["statistics"]["minimum"] = self.raster_np[index].min()
                stats[band]["statistics"]["maximum"] = self.raster_np[index].max()
                stats[band]["statistics"]["mean"] = self.raster_np[index].mean()
                stats[band]["statistics"]["median"] = np.median(self.raster_np[index])
                stats[band]["statistics"]["std"] = self.raster_np[index].std()
                stats[band]["histogram"]["buckets"] = list(np.bincount(self.raster_np.flatten()))
        mean_minimum = np.mean([band_stat["statistics"]["minimum"] for band_stat in stats.values()])
        mean_maximum = np.mean([band_stat["statistics"]["maximum"] for band_stat in stats.values()])
        mean_mean = np.mean([band_stat["statistics"]["mean"] for band_stat in stats.values()])
        mean_median = np.mean([band_stat["statistics"]["median"] for band_stat in stats.values()])
        mean_std = np.mean([band_stat["statistics"]["std"] for band_stat in stats.values()])
        hists_np = [np.asarray(band_stat["histogram"]["buckets"]) for band_stat in stats.values()]
        mean_hist_np = np.sum(hists_np, axis=0) / len(hists_np)
        mean_hist = list(mean_hist_np.astype(int))
        stats["all"] = {
            "statistics": {"minimum": mean_minimum, "maximum": mean_maximum, "mean": mean_mean,
                           "median": mean_median, "std": mean_std},
            "histogram": {"buckets": mean_hist}}
        self.close_raster()
        return stats

    def write_multiband_from_singleband_rasters_as_vrt(self, out_dir: Union[str, Path] = None):
        """Writes a multiband raster to file from a pre-built VRT. For debugging and demoing"""
        out_dir = self.root_dir

        if out_dir is None:
            logging.error(f"There is no path for the output, root_dir shoudn't be None")
            return
        if not self.raster.driver == 'VRT':
            logging.error(f"To write a multi-band raster from single-band files, a VRT must be provided."
                          f"\nGot {self.raster.meta}")
            return
        if "${dataset.bands}" in self.raster_raw_input:
            out_tif_path = out_dir / Path(self.raster_raw_input).name.replace("${dataset.bands}", ''.join(self.raster_bands_request))
        elif is_stac_item(self.raster_raw_input):
            out_tif_path = out_dir / f"{Path(self.raster_raw_input).stem}_{'-'.join(self.raster_bands_request)}.tif"
        else:
            logging.error(f"\nTo write multiband raster from single band imagery, "
                          f"source imagery must be referenced with expected formats.\n"
                          f"See dataset/README.md")
            return
        logging.debug(f"Writing multi-band raster to {out_tif_path}")
        create_new_raster_from_base(
            input_raster=self.raster,
            output_raster=str(out_tif_path),
            write_array=self.raster.read())
        return out_tif_path

    @staticmethod
    def parse_input_raster(
            csv_raster_str: str,
            raster_bands_requested: Sequence
    ) -> Union[List, Tuple]:
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
            item = SingleBandItemEO(item=pystac.Item.from_file(csv_raster_str),
                                    bands_requested=raster_bands_requested)
            raster = [value['meta'].href for value in item.bands_requested.values()]
            return raster
        elif "${dataset.bands}" in csv_raster_str:
            if not isinstance(raster_bands_requested, (List, ListConfig, tuple)) or len(raster_bands_requested) == 0:
                raise TypeError(f"\nRequested bands should a list of bands. "
                                f"\nGot {raster_bands_requested} of type {type(raster_bands_requested)}")
            raster = [csv_raster_str.replace("${dataset.bands}", band) for band in raster_bands_requested]
            return raster
        else:
            try:
                validate_raster(csv_raster_str)
                return [csv_raster_str]
            except (FileNotFoundError, rasterio.RasterioIOError, TypeError) as e:
                logging.critical(f"Couldn't parse input raster. Got {csv_raster_str}")
                raise e

    @staticmethod
    def bounds_iou(polygon1: Polygon, polygon2: Polygon) -> float:
        """Calculate intersection over union of areas between two shapely polygons"""
        if not polygon1.intersects(polygon2):
            return 0
        else:
            intersection = polygon1.intersection(polygon2).area
            union = polygon1.area + polygon2.area - intersection
            return intersection / union

    @staticmethod
    def bounds_iou_gdf_riodataset(gdf: gpd.GeoDataFrame, raster: rasterio.DatasetReader) -> float:
        """Calculates intersection over union of the total bounds of a GeoDataFrame and bounds of a rasterio Dataset"""
        label_bounds = gdf.total_bounds
        label_bounds_box = box(*label_bounds.tolist())
        raster_bounds_box = box(*list(raster.bounds))
        bounds_iou = AOI.bounds_iou(polygon1=label_bounds_box, polygon2=raster_bounds_box)
        return bounds_iou

    @staticmethod
    def filter_gdf_by_attribute(
            gdf_tile: Union[str, Path, gpd.GeoDataFrame],
            attr_field: str = None,
            attr_vals: Sequence = None):
        """
        Filter features from a geopandas.GeoDataFrame according to an attribute field and filtering values
        @param gdf_tile: str, Path or gpd.GeoDataFrame
            GeoDataFrame or path to GeoDataFrame to filter feature from
        @return: Subset of source GeoDataFrame with only filtered features (deep copy)
        """
        gdf_tile = _check_gdf_load(gdf_tile)
        if gdf_tile.empty or not attr_field or not attr_vals:
            return gdf_tile
        try:
            condList = [gdf_tile[f'{attr_field}'] == val for val in attr_vals]
            condList.extend([gdf_tile[f'{attr_field}'] == str(val) for val in attr_vals])
            allcond = functools.reduce(lambda x, y: x | y, condList)  # combine all conditions with OR
            gdf_filtered = gdf_tile[allcond].copy(deep=True)
            logging.debug(f'Successfully filtered features from GeoDataFrame"\n'
                          f'Filtered features: {len(gdf_filtered)}\n'
                          f'Total features: {len(gdf_tile)}\n'
                          f'Attribute field: "{attr_field}"\n'
                          f'Filtered values: {attr_vals}')
            return gdf_filtered
        except KeyError as e:
            logging.critical(f'No attribute named {attr_field} in GeoDataFrame. \n'
                             f'If all geometries should be kept, leave "attr_field" and "attr_vals" blank.\n'
                             f'Attributes: {gdf_tile.columns}\n'
                             f'GeoDataFrame: {gdf_tile.info()}')
            raise e

    def close_raster(self) -> None:
        if self.raster_closed is False:
            self.raster.close()
            self.raster_closed = True


def aois_from_csv(
        csv_path: Union[str, Path],
        bands_requested: List = None,
        attr_field_filter: str = None,
        attr_values_filter: str = None,
        download_data: bool = False,
        data_dir: str = "data",
        for_multiprocessing = False,
):
    """
    Creates list of AOIs by parsing a csv file referencing input data
    @param csv_path:
        path to csv file containing list of input data. See README for details on expected structure of csv.
    @param bands_requested:
        List of bands to select from inputted imagery. Applies only to single-band input imagery.
    @param attr_values_filter:
        Attribute filed to filter features from
    @param attr_field_filter:
        Attribute values (for given attribute field) for features to keep
    @param download_data:
        if True, download dataset and store it in the root directory.
    @param data_dir:
        root directory where data can be found or downloaded
    Returns: a list of AOIs objects
    """
    aois = []
    data_list = read_csv(csv_path)
    logging.info(f'\n\tSuccessfully read csv file: {Path(csv_path).name}\n'
                 f'\tNumber of rows: {len(data_list)}\n'
                 f'\tCopying first row:\n{data_list[0]}\n')
    for i, aoi_dict in tqdm(enumerate(data_list), desc="Creating AOI's"):
        try:
            new_aoi = AOI.from_dict(
                aoi_dict=aoi_dict,
                bands_requested=bands_requested,
                attr_field_filter=attr_field_filter,
                attr_values_filter=attr_values_filter,
                download_data=download_data,
                root_dir=data_dir,
                for_multiprocessing=for_multiprocessing,
            )
            logging.debug(new_aoi)
            aois.append(new_aoi)
        except FileNotFoundError as e:
            logging.critical(f"{e}\nGround truth file may not exist or is empty.\n"
                             f"Failed to create AOI:\n{aoi_dict}\n"
                             f"Index: {i}")
    return aois
