import functools
import json
from typing import Union, Sequence, Tuple, List, Dict
import geopandas as gpd
import numpy as np
import pyproj
import pystac
import rasterio
import dask.array as da
from skimage.exposure import equalize_adapthist
from scipy.special import expit
import torch
from torch.nn import functional as F
from pathlib import Path
import sys
import dask
from hydra.utils import to_absolute_path
from pandas.io.common import is_url
from omegaconf import listconfig, ListConfig
from tqdm import tqdm
from rasterio.windows import from_bounds
from numba import cuda
import scipy.signal.windows as w

if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from dataset.stacitem import SingleBandItemEO
from utils.geoutils import (
    is_stac_item,
    check_gdf_load,
    bounds_gdf,
    bounds_riodataset,
    overlap_poly1_rto_poly2,
    gdf_mean_vertices_nb,
)
from utils.logger import get_logger
from utils.utils import read_csv, minmax_scale, download_url_wcheck
from utils.verifications import (
    assert_crs_match,
    validate_raster,
    validate_raster_dask,
    validate_features_from_gpkg,
)

logging = get_logger(__name__)  # import logging


class AOI(object):
    """
    Object containing all data information about a single area of interest
    based on https://github.com/stac-extensions/ml-aoi
    """

    def __init__(
        self,
        raster: Union[Path, str],
        raster_bands_request: List = [],
        label: Union[Path, str] = None,
        split: str = None,
        aoi_id: str = None,
        collection: str = None,
        raster_num_bands_expected: int = None,
        attr_field_filter: str = None,
        attr_values_filter: Sequence = None,
        root_dir: str = "data",
        for_multiprocessing: bool = False,
        raster_stats: bool = False,
        equalize_clahe_clip_limit: float = 0,
        chunk_size: int = 1024,
        rio: str = None,
    ):
        """
        @param raster: pathlib.Path or str
            Path to source imagery
        @param raster_bands_request: list
            list of bands
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
        @param root_dir:
            root directory where dataset can be found or downloaded
        @param for_multiprocessing: bool, optional
            If True, no rasterio.DatasetReader will be generated in __init__. User will have to call open raster later.
            See: https://github.com/rasterio/rasterio/issues/1731
        @param raster_stats:
            if True, radiometric stats will be read from Stac Item if available or calculated
        @param equalize_clahe_clip_limit: int, optional
            Threshold value for contrast limiting. If 0 clipping is disabled.
            Geo-deep-learning enforces the use of an integer to avoid confusion with sklearn's CLAHE algorithm, which
            expects float between 0 and 1. See:
            https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
            https://kornia.readthedocs.io/en/latest/enhance.html#kornia.enhance.equalize_clahe
        @parem chunk_size: int,optional
            The chunk size for chunking the dask array
        """

        self.raster_name = Path(raster)  # default name, may be overwritten later
        self.label = None
        self.label_gdf = None
        self.label_invalid_features = None
        self.label_bounds = None
        self.overlap_label_rto_raster = self.overlap_raster_rto_label = None
        self.epsg_raster = self.epsg_label = None
        self.crs_match = None

        """ ---------------------------Input Validation-------------------------------"""
        if not isinstance(raster, str):
            raise TypeError(
                f"Raster path should be a string.\nGot {raster} of type {type(raster)}"
            )
        self.raster_raw_input = raster

        if raster_bands_request and not isinstance(
            raster_bands_request, (Sequence, ListConfig)
        ):
            raise ValueError(
                f"Requested bands should be a list."
                f"\nGot {raster_bands_request} of type {type(raster_bands_request)}"
            )
        self.raster_bands_request = raster_bands_request

        if not isinstance(equalize_clahe_clip_limit, float):
            raise ValueError(
                f"Enhance clip limit should be a float. See documentation.\n"
                f"Got {type(equalize_clahe_clip_limit)}."
            )
        if equalize_clahe_clip_limit > 1 or equalize_clahe_clip_limit < 0:
            raise ValueError(
                f"Enhance clip limit be a float between 0 and 1, inclusive.\n"
                f"Got {type(equalize_clahe_clip_limit)}."
            )

        self.enhance_clip_limit = equalize_clahe_clip_limit

        if split and not isinstance(split, str):
            raise ValueError(f"\nDataset split should be a string.\nGot {split}.")

        if label and split not in ["trn", "tst", "inference"]:
            raise ValueError(
                f"\nWith ground truth, split should be 'trn', 'tst' or 'inference'. \nGot {split}"
            )
        # force inference split if no label provided
        elif not label and (split != "inference" or not split):
            logging.warning(
                f"\nNo ground truth provided. Dataset split will be set to 'inference'"
                f"\nOriginal split: {split}"
            )
            split = "inference"
        self.split = split

        if aoi_id and not isinstance(aoi_id, str):
            raise TypeError(
                f"AOI name should be a string. Got {aoi_id} of type {type(aoi_id)}"
            )
        elif not aoi_id:
            aoi_id = self.raster_name.stem  # Defaults to name of image without suffix
        self.aoi_id = aoi_id

        if collection and not isinstance(collection, str):
            raise TypeError(
                f"Collection name should be a string. Got {collection} of type {type(collection)}"
            )
        self.aoi_id = aoi_id

        # If ground truth is provided, check attribute field
        if label and attr_field_filter:
            if not isinstance(attr_field_filter, str):
                raise TypeError(
                    f"Attribute field name should be a string.\n"
                    f"Got {attr_field_filter} of type {type(attr_field_filter)}"
                )
            elif attr_field_filter not in self.label_gdf.columns:
                # fiona and geopandas don't expect attribute name exactly the same way: "properties/class" vs "class"
                attr_field_filter = attr_field_filter.split("/")[-1]
                if attr_field_filter not in self.label_gdf.columns:
                    raise ValueError(
                        f'\nAttribute field "{attr_field_filter}" not found in label attributes:\n'
                        f"{self.label_gdf.columns}"
                    )
        self.attr_field_filter = attr_field_filter

        # If ground truth is provided, check attribute values to filter from
        if isinstance(attr_values_filter, int):
            attr_values_filter = [attr_values_filter]
        if (
            label
            and attr_values_filter
            and not isinstance(attr_values_filter, (list, listconfig.ListConfig))
        ):
            raise TypeError(
                f"Attribute values should be a list.\n"
                f"Got {attr_values_filter} of type {type(attr_values_filter)}"
            )
        self.attr_values_filter = attr_values_filter
        if not isinstance(chunk_size, int):
            raise ValueError(
                f"\n chunk_size should be an interger \n Got {chunk_size}."
            )
        self.chunk_size = chunk_size
        if not isinstance(for_multiprocessing, bool):
            raise ValueError(
                f'\n"for_multiprocessing" should be a boolean.\nGot {for_multiprocessing}.'
            )

        self.for_multiprocessing = for_multiprocessing
        self.root_dir = Path(root_dir)

        """ -------------------------------------------------------------------"""

        """ -------------------------Processing---------------------------------"""
        # it constructs a list of URLs or file paths for raster bands: This function now returns a dict of {band:url}
        raster_parsed = self.parse_input_raster(
            csv_raster_str=self.raster_raw_input,
            raster_bands_requested=self.raster_bands_request,
        )

        logging.info(f"\n\tSuccessfully parsed Rasters \n: {raster_parsed}\n")
        # If stac item input, keep Stac item object as attribute
        if is_stac_item(self.raster_raw_input):
            item = SingleBandItemEO(
                item=pystac.Item.from_file(self.raster_raw_input),
                bands_requested=self.raster_bands_request,
            )
            self.stack_item = pystac.Item.from_file(self.raster_raw_input)
            self.raster_stac_item = item
            self.raster_stats = self.read_stack_stat() if raster_stats else {}
        else:
            self.raster_stac_item = None
            self.raster_stats = {}
        # update the raster_src_is_multiband property
        self.raster_src_is_multiband = False
        if len(raster_parsed) == 1:
            raster_count = rasterio.open(next(iter(raster_parsed.values()))).count
            if raster_count > 1:
                self.raster_src_is_multiband = True
        else:
            if len(raster_parsed) == 3:
                desired_bands = (
                    ["R", "G", "B"]
                    if not self.raster_stac_item
                    else ["red", "green", "blue"]
                )
            elif len(raster_parsed) == 4:
                desired_bands = (
                    ["N", "R", "G", "B"]
                    if not self.raster_stac_item
                    else ["nir", "red", "green", "blue"]
                )
            # Create a new dictionary with the desired key order
            raster_parsed = {
                key: raster_parsed[key]
                for key in desired_bands
                if key in self.raster_bands_request
            }
            # TODO: how to handle Nir?

        # Chack the size of tiff files
        for single_raster in raster_parsed.values():
            size = (
                Path(single_raster).stat().st_size
                if not is_url(single_raster)
                else None
            )
            logging.debug(
                f"Raster to validate: {raster}\n"
                f"Size: {size}\n"
                f"Extended check: {False}"
            )
        self.raster_parsed = raster_parsed
        if label:
            self.label = Path(label)
            self.label_gdf = check_gdf_load(label)
            self.label_invalid_features = validate_features_from_gpkg(label)
        rio_gdal_options = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        }
        stat_calculated = False if len(self.raster_stats) == 0 else True
        if raster_stats and not stat_calculated:
            self.raster_stats = {name: {} for name in self.raster_bands_request}

        np_sources = []
        with rasterio.Env(**rio_gdal_options):
            for tiff_band, raster in self.raster_parsed.items():
                with rasterio.open(raster, "r") as src:
                    # Getting the bbox window
                    self.raster = src
                    self.raster_meta = self.raster.meta
                    if rio is not None:
                        bbox = tuple(map(float, rio.split(", ")))
                        self.roi_window = from_bounds(
                            left=bbox[0],
                            bottom=bbox[1],
                            right=bbox[2],
                            top=bbox[3],
                            transform=src.transform,
                        )
                    # Label check
                    self.raster_bounds = bounds_riodataset(src)
                    if label:
                        self.label_bounds = bounds_gdf(self.label_gdf)
                        if not self.raster_bounds.intersects(self.label_bounds):
                            logging.error(
                                f"Features in label file {label} do not intersect with bounds of raster file "
                                f"{self.raster.name}"
                            )
                        self.overlap_label_rto_raster = overlap_poly1_rto_poly2(
                            self.label_bounds, self.raster_bounds
                        )
                        self.overlap_raster_rto_label = overlap_poly1_rto_poly2(
                            self.raster_bounds, self.label_bounds
                        )

                        # TODO: unit test for failed CRS match
                        try:
                            # TODO: check if this creates overhead. Skip if report exists?
                            self.crs_match, self.epsg_raster, self.epsg_label = (
                                assert_crs_match(self.raster, self.label_gdf)
                            )
                        except pyproj.exceptions.CRSError as e:
                            logging.warning(
                                f"\nError while checking CRS match between raster and label."
                                f"\n{e}"
                            )

                    # validate each raster
                    validate_raster_dask(self.raster)
                    bands_count = range(1, src.count + 1)
                    # Just to make sure that if we have a subset of data, get the subset only
                    if self.raster_src_is_multiband and self.raster_bands_request:
                        bands_count = self.raster_bands_request
                    # prepping the dict for stat, if it is a multi-band and stat is not calculated before (not stack)
                    if (
                        self.raster_src_is_multiband
                        and raster_stats
                        and not stat_calculated
                    ):
                        self.raster_stats = {
                            f"band_{index}": {} for index in bands_count
                        }

                    # Iterating over bands
                    for band in bands_count:
                        self.raster = (
                            self.raster.read(band)
                            if not rio
                            else self.raster.read(indexes=band, window=self.roi_window)
                        )
                        if self.raster.dtype == np.uint16:
                            self.raster = self.raster.astype(np.int32)
                        elif self.raster.dtype == np.uint32:
                            self.raster = self.raster.astype(np.int64)
                        # calculating raster stat for each band
                        if raster_stats and not stat_calculated:
                            self.raster_stats = self.calc_raster_stats(
                                "band_" + str(band)
                                if self.raster_src_is_multiband
                                else tiff_band
                            )
                        self.high_or_low_contrast = self.is_low_contrast()
                        np_sources.append(self.raster)
                        self.raster = src if self.raster_src_is_multiband else None

        # calculating raster stat for all bands :: Only for multi-bands, otherwise, we'll get error on not having the same bucket size for all bands
        if raster_stats and self.raster_src_is_multiband and not stat_calculated:
            self.raster_stats = self.calc_rasters_stats()
        """ Create a all_bands array to store all RGB bands; This property will be deleted as soon as we create a dask array in out dask cluster; 
            so it doesn't occupy memory ultimately """
        self.all_bands_array = np.array(np_sources)
        self.num_bands = self.all_bands_array.shape[0]

        if raster_num_bands_expected:
            if not len(np_sources) == raster_num_bands_expected:
                logging.critical(
                    f"The number of bands expected doesn't match number of bands in input image.\n"
                    f"Expected: {raster_num_bands_expected} bands\n"
                    f"Got: {len(bands_count)} bands\n"
                    f"Raster path: {raster.name}"
                )
                raise ValueError()

        if label:
            self.label_gdf_filtered = self.filter_gdf_by_attribute(
                self.label_gdf.copy(deep=True),
                self.attr_field_filter,
                self.attr_values_filter,
            )
            if len(self.label_gdf_filtered) == 0:
                logging.warning(
                    f'\nNo features found for ground truth "{self.label}",'
                    f'\nfiltered by attribute field "{self.attr_field_filter}"'
                    f'\nwith values "{self.attr_values_filter}"'
                )
        else:
            self.label_gdf_filtered = None

        logging.debug(self)

    @classmethod
    def from_dict(
        cls,
        aoi_dict,
        bands_requested: List = None,
        attr_field_filter: str = None,
        attr_values_filter: list = None,
        root_dir: str = "data",
        raster_stats=False,
        for_multiprocessing: bool = False,
        equalize_clahe_clip_limit: int = 0,
        chunk_size: int = 512,
    ):
        """Instanciates an AOI object from an input-data dictionary as expected by geo-deep-learning"""
        if not isinstance(aoi_dict, dict):
            raise TypeError("Input data should be a dictionary.")
        if not {"tif", "gpkg", "split"}.issubset(set(aoi_dict.keys())):
            raise ValueError(
                "Input data should minimally contain the following keys: \n"
                "'tif', 'gpkg', 'split'."
            )
        if aoi_dict["gpkg"] is None:
            logging.warning(
                f"No ground truth data found for {aoi_dict['tif']}.\n"
                f"Only imagery will be processed from now on"
            )
        if "aoi_id" not in aoi_dict.keys() or not aoi_dict["aoi_id"]:
            aoi_dict["aoi_id"] = Path(aoi_dict["tif"]).stem
        aoi_dict["attribute_name"] = attr_field_filter
        new_aoi = cls(
            raster=aoi_dict["tif"],
            raster_bands_request=bands_requested,
            label=aoi_dict["gpkg"],
            split=aoi_dict["split"],
            attr_field_filter=attr_field_filter,
            attr_values_filter=attr_values_filter,
            aoi_id=aoi_dict["aoi_id"],
            for_multiprocessing=for_multiprocessing,
            raster_stats=raster_stats,
            equalize_clahe_clip_limit=equalize_clahe_clip_limit,
            chunk_size=chunk_size,
        )
        return new_aoi

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

    def to_dict(self, extended=True):
        """returns a dictionary containing all important attributes of AOI (ex.: to print a report or output csv)"""
        try:
            raster_area = (self.raster.res[0] * self.raster.width) * (
                self.raster.res[1] * self.raster.height
            )
        except AttributeError:
            raster_area = None
        out_dict = {
            "raster": self.raster_raw_input,
            "label": self.label,
            "split": self.split,
            "id": self.aoi_id,
            "raster_parsed": self.raster_parsed,
            "raster_area": raster_area,
            "raster_meta": repr(self.raster_meta)
            .replace("\n", "")
            .replace(" ", " ")
            .replace(",", ";"),
            "label_features_nb": len(self.label_gdf),
            "label_features_filtered_nb": len(self.label_gdf_filtered),
            "overlap_label_rto_raster": self.overlap_label_rto_raster,
            "overlap_raster_rto_label": self.overlap_raster_rto_label,
            "crs_raster": self.epsg_raster,
            "crs_label": self.epsg_label,
            "crs_match": self.crs_match,
        }
        if extended:
            mean_ext_vert_nb = gdf_mean_vertices_nb(self.label_gdf_filtered)

            out_dict.update(
                {
                    "label_features_filtered_mean_area": np.mean(
                        self.label_gdf_filtered.area
                    ),
                    "label_features_filtered_mean_perimeter": np.mean(
                        self.label_gdf_filtered.length
                    ),
                    "label_features_filtered_mean_exterior_vertices_nb": mean_ext_vert_nb,
                }
            )
        return out_dict

    def read_stack_stat(self):
        """For stac items formatted as expected, reads mean and std of raster imagery, per band.
        See stac item example: tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03.json"""

        if self.raster_bands_request:
            stats = {name: {} for name in self.raster_bands_request}
        else:
            stats = {f"band_{index}": {} for index in range(self.raster.count)}
        try:
            stats_asset = self.raster_stac_item.item.assets["STATS"]
            # download can be changed to streaming using request
            if is_url(stats_asset.href):
                """
                response = requests.get(stats_asset.href)
                response.raise_for_status()  # Ensure we raise an error for bad responses
                stac_stats = response.json()  # Directly parse the JSON response
                """
                download_url_wcheck(
                    stats_asset.href,
                    root=str(self.root_dir),
                    filename=Path(stats_asset.href).name,
                )
                stats_href = self.root_dir / Path(stats_asset.href).name
            else:
                stats_href = to_absolute_path(stats_asset.href)
            with open(stats_href, "r") as ifile:
                stac_stats = json.loads(ifile.read())
            stac_stats = {
                bandwise_stats["asset"]: bandwise_stats for bandwise_stats in stac_stats
            }
            for band in self.raster_stac_item.bands:
                stats[band.common_name] = stac_stats[band.name]
            return stats
        except (AttributeError, KeyError):
            return {}

    def calc_raster_stats(self, band):
        """If source imagery is not a stac item or stac item lacks stats assets, stats are calculed on the fly"""

        stats = self.raster_stats
        stats[band] = {"statistics": {}, "histogram": {}}
        stats[band]["statistics"]["minimum"] = self.raster.min()
        stats[band]["statistics"]["maximum"] = self.raster.max()
        stats[band]["statistics"]["mean"] = self.raster.mean()
        stats[band]["statistics"]["median"] = np.median(self.raster)
        stats[band]["statistics"]["std"] = self.raster.std()
        stats[band]["histogram"]["buckets"] = list(np.bincount(self.raster.flatten()))
        return stats

    def calc_rasters_stats(self):
        """If source imagery is not a stac item or stac item lacks stats assets, stats are calculed on the fly"""
        stats = self.raster_stats
        mean_minimum = np.mean(
            [band_stat["statistics"]["minimum"] for band_stat in stats.values()]
        )
        mean_maximum = np.mean(
            [band_stat["statistics"]["maximum"] for band_stat in stats.values()]
        )
        mean_mean = np.mean(
            [band_stat["statistics"]["mean"] for band_stat in stats.values()]
        )
        mean_median = np.mean(
            [band_stat["statistics"]["median"] for band_stat in stats.values()]
        )
        mean_std = np.mean(
            [band_stat["statistics"]["std"] for band_stat in stats.values()]
        )
        hists_np = [
            np.asarray(band_stat["histogram"]["buckets"])
            for band_stat in stats.values()
        ]
        mean_hist_np = np.sum(hists_np, axis=0) / len(hists_np)
        mean_hist = list(mean_hist_np.astype(int))
        stats["all"] = {
            "statistics": {
                "minimum": mean_minimum,
                "maximum": mean_maximum,
                "mean": mean_mean,
                "median": mean_median,
                "std": mean_std,
                "low_contrast": self.high_or_low_contrast,
            },
            "histogram": {"buckets": mean_hist},
        }
        return stats

    def create_dask_array(self):
        aoi_dask_array = da.from_array(
            self.all_bands_array,
            chunks=(
                1,
                int(self.chunk_size / 2),
                int(self.chunk_size / 2),
            ),
        )
        del self.all_bands_array
        return aoi_dask_array

    def read_raster_chunk(url, window):
        with rasterio.open("/vsicurl/" + url) as src:
            return src.read(1, window=window)

    @staticmethod
    def equalize_adapthist_enhancement(aoi_chunk: np.ndarray, clip_limit: float):
        """This function applies scikit image equalize_adapthist on each chunk of dask array
        each chunk is [band, height, width] --> So, the contrast enhancement is applied on each hand separately"""
        from kornia import image_to_tensor, tensor_to_image
        from kornia.enhance import equalize_clahe

        if aoi_chunk.size > 0 and aoi_chunk is not None:
            ready_np_array = aoi_chunk[0, :, :]
            ready_np_array = minmax_scale(
                img=ready_np_array, scale_range=(0, 1), orig_range=(0, 255)
            )
            ready_np_array = image_to_tensor(ready_np_array)
            """ ready_np_array = equalize_adapthist(
                ready_np_array,
                clip_limit=clip_limit,
            )"""
            clip_limit = 25
            ready_np_array = equalize_clahe(
                ready_np_array.float().unsqueeze(0), clip_limit=float(clip_limit)
            )
            ready_np_array = tensor_to_image(
                (ready_np_array * 255).long(), keepdim=True
            ).squeeze()
            # ready_np_array = (ready_np_array * 255).astype(np.int32)
            return ready_np_array[None, :, :]
        return aoi_chunk  # If the chunk size is 0, return the original chunk

    @staticmethod
    def sum_overlapped_chunks(
        aoi_chunk: np.ndarray,
        chunk_size: int,
        block_info=None,
    ):
        num_chunks = block_info[0]["num-chunks"]
        chunk_location = block_info[0]["chunk-location"]
        full_array = np.empty((1, 1))

        if aoi_chunk.size > 0 and aoi_chunk is not None:
            if (chunk_location[1] == 0 or chunk_location[1] == num_chunks[1] - 1) and (
                chunk_location[2] == 0 or chunk_location[2] == num_chunks[2] - 1
            ):
                """ All 4 corners"""
                full_array = aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    : int(chunk_size / 2),
                ]
            elif (
                chunk_location[1] == 0 or chunk_location[1] == num_chunks[1] - 1
            ) and (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1):
                """ Top and bottom edges but not corners"""
                full_array = (
                    aoi_chunk[
                        :,
                        : int(chunk_size / 2),
                        int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    ]
                    + aoi_chunk[
                        :,
                        : int(chunk_size / 2),
                        : int(chunk_size / 2),
                    ]
                )
            elif (
                chunk_location[2] == 0 or chunk_location[2] == num_chunks[2] - 1
            ) and (chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1):
                """ Left and right edges but not corners"""
                full_array = (
                    aoi_chunk[
                        :,
                        int(chunk_size / 2) : int(chunk_size / 2) * 2,
                        : int(chunk_size / 2),
                    ]
                    + aoi_chunk[
                        :,
                        : int(chunk_size / 2),
                        : int(chunk_size / 2),
                    ]
                )
            elif (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1) and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
            ):
                """ Middle chunks """
                full_array = (
                    aoi_chunk[
                        :,
                        : int(chunk_size / 2),
                        : int(chunk_size / 2),
                    ]
                    + aoi_chunk[
                        :,
                        : int(chunk_size / 2),
                        int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    ]
                    + aoi_chunk[
                        :,
                        int(chunk_size / 2) : int(chunk_size / 2) * 2,
                        : int(chunk_size / 2),
                    ]
                    + aoi_chunk[
                        :,
                        int(chunk_size / 2) : int(chunk_size / 2) * 2,
                        int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    ]
                )

            if full_array.shape != (
                aoi_chunk.shape[0],
                int(chunk_size / 2),
                int(chunk_size / 2),
            ):
                logging.error(
                    f" In sum_overlapped_chunks the shape of full_array is not {(6, int(chunk_size / 2), int(chunk_size / 2))}"
                    f" The size of it {full_array.shape}"
                )
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    final_result = np.divide(
                        full_array[:-1, :, :],
                        full_array[-1, :, :][np.newaxis, :, :],
                        out=np.zeros_like(full_array[:-1, :, :], dtype=float),
                        where=full_array[-1, :, :] != 0,
                    )
                    if final_result.shape[0] == 1:
                        final_result = expit(final_result)
                        final_result = (
                            np.where(final_result > 0.5, 1, 0)
                            .squeeze(0)
                            .astype(np.uint8)
                        )
                    else:
                        final_result = np.argmax(final_result, axis=0).astype(np.uint8)
                    return final_result

    def is_low_contrast(self, fraction_threshold=0.3):
        """This function checks if a raster is low contrast
        https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.is_low_contrast
        Args:
            fraction_threshold (float, optional): low contrast fraction threshold. Defaults to 0.3.
        Returns:
            bool: False for high contrast image | True for low contrast image
        """
        data_type = self.raster.dtype
        grayscale = np.mean(self.raster, axis=0)
        grayscale = np.round(grayscale).astype(data_type)
        from skimage import exposure

        high_or_low_contrast = exposure.is_low_contrast(
            grayscale, fraction_threshold=fraction_threshold
        )
        return high_or_low_contrast

    @staticmethod
    def parse_input_raster(
        csv_raster_str: str, raster_bands_requested: Sequence
    ) -> Union[Dict, Tuple]:
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
        raster = {}
        if is_stac_item(csv_raster_str):
            item = SingleBandItemEO(
                item=pystac.Item.from_file(csv_raster_str),
                bands_requested=raster_bands_requested,
            )
            for key, value in item.bands_requested.items():
                raster[key] = value["meta"].href
            return raster
        elif "${dataset.bands}" in csv_raster_str:
            if (
                not raster_bands_requested
                or not isinstance(raster_bands_requested, (List, ListConfig, tuple))
                or len(raster_bands_requested) == 0
            ):
                raise TypeError(
                    f"\nRequested bands should be a list of bands. "
                    f"\nGot {raster_bands_requested} of type {type(raster_bands_requested)}"
                )
            for band in raster_bands_requested:
                raster[band] = csv_raster_str.replace("${dataset.bands}", str(band))
            return raster
        else:
            try:
                validate_raster(csv_raster_str)
                return {"all_counts": csv_raster_str}
            except (FileNotFoundError, rasterio.RasterioIOError, TypeError) as e:
                logging.critical(f"Couldn't parse input raster. Got {csv_raster_str}")
                raise e

    @staticmethod
    def filter_gdf_by_attribute(
        gdf_patch: Union[str, Path, gpd.GeoDataFrame],
        attr_field: str = None,
        attr_vals: Sequence = None,
    ):
        """
        Filter features from a geopandas.GeoDataFrame according to an attribute field and filtering values
        @param gdf_patch: str, Path or gpd.GeoDataFrame
            GeoDataFrame or path to GeoDataFrame to filter feature from
        @return: Subset of source GeoDataFrame with only filtered features (deep copy)
        """
        gdf_patch = check_gdf_load(gdf_patch)
        if gdf_patch.empty or not attr_field or not attr_vals:
            return gdf_patch
        try:
            condList = [gdf_patch[f"{attr_field}"] == val for val in attr_vals]
            condList.extend(
                [gdf_patch[f"{attr_field}"] == str(val) for val in attr_vals]
            )
            allcond = functools.reduce(
                lambda x, y: x | y, condList
            )  # combine all conditions with OR
            gdf_filtered = gdf_patch[allcond].copy(deep=True)
            logging.debug(
                f'Successfully filtered features from GeoDataFrame"\n'
                f"Filtered features: {len(gdf_filtered)}\n"
                f"Total features: {len(gdf_patch)}\n"
                f'Attribute field: "{attr_field}"\n'
                f"Filtered values: {attr_vals}"
            )
            return gdf_filtered
        except KeyError as e:
            logging.critical(
                f"No attribute named {attr_field} in GeoDataFrame. \n"
                f'If all geometries should be kept, leave "attr_field" and "attr_vals" blank.\n'
                f"Attributes: {gdf_patch.columns}\n"
                f"GeoDataFrame: {gdf_patch.info()}"
            )
            raise e

    @staticmethod
    def runModel(
        chunk_data: np.ndarray,
        chunk_size: int,
        model_path: str,
        block_info=None,
    ):
        num_chunks = block_info[0]["num-chunks"]
        chunk_location = block_info[0]["chunk-location"]

        if chunk_data.size > 0 and chunk_data is not None:
            try:
                # Defining the base window for window creation later
                step = chunk_size >> 1
                window = w.hann(M=chunk_size, sym=False)
                window = window[:, np.newaxis] * window[np.newaxis, :]
                final_window = np.empty((1, 1))

                # device = torch.device("cpu")
                device_id = None
                num_devices = 0
                if torch.cuda.is_available():
                    num_devices = torch.cuda.device_count()
                    for i in range(num_devices):
                        res = {"gpu": torch.cuda.utilization(i)}
                        torch_cuda_mem = torch.cuda.mem_get_info(i)
                        mem = {
                            "used": torch_cuda_mem[-1] - torch_cuda_mem[0],
                            "total": torch_cuda_mem[-1],
                        }
                        used_ram = mem["used"] / (1024**2)
                        max_ram = mem["total"] / (1024**2)
                        used_ram_percentage = (used_ram / max_ram) * 100
                        if used_ram_percentage < 70 and res["gpu"] < 70:
                            device_id = i
                            break
                device = (
                    torch.device(f"cuda:{device_id}")
                    if device_id is not None
                    else torch.device("cpu")
                )
                device = torch.device("cuda:0")
                """ 
                if (
                    chunk_location[2] > int(num_chunks[2] / num_devices)
                    and chunk_location[2] <= int(num_chunks[2] / num_devices) * 2
                ):
                    device = torch.device("cuda:1")
                elif chunk_location[2] > int(num_chunks[2] / num_devices) * 2:
                    device = torch.device("cuda:2")
                """
                model = torch.jit.load(model_path, map_location=device).to(
                    device
                )  # load the model
                tensor = torch.as_tensor(chunk_data[np.newaxis, ...]).to(
                    model.device
                )  # convert chunck to torch Tensor
                prediction_output = np.empty(
                    shape=(5, chunk_data.shape[1], chunk_data.shape[2])
                )  # Create the output but empty

                with torch.no_grad():
                    prediction_output = model(tensor).cpu().numpy()[0]  # run the model
                del tensor
                del model  # Delete the model if it's not needed anymore

                # Getting a (chunk_size, chunk_size) chunk with its window
                if chunk_location[2] == num_chunks[2] - 1 and chunk_location[1] == 0:
                    """ If chunk is top right corner"""
                    prediction_output = prediction_output[
                        :,
                        :chunk_size,
                        :chunk_size,
                    ]
                    """ Top right corner window"""
                    window_u = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_u[:step, :step], np.ones((step, step))],
                            [window_r[step:, :step], window_r[step:, step:]],
                        ]
                    )
                elif chunk_location[2] == num_chunks[2] - 1 and (
                    chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
                ):
                    """ If chunk is on the right egde but not corners"""
                    prediction_output = prediction_output[
                        :,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                        :chunk_size,
                    ]
                    """ left egde window """
                    final_window = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                elif chunk_location[2] == num_chunks[2] - 1 and (
                    chunk_location[1] == num_chunks[1] - 1
                ):
                    """ If chunk is in the bottom right corner"""
                    prediction_output = prediction_output[
                        :,
                        :chunk_size,
                        :chunk_size,
                    ]
                    """ bottom right window """
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_r[:step, :step], window_r[:step, step:]],
                            [window_b[step:, :step], np.ones((step, step))],
                        ]
                    )
                elif chunk_location[1] == num_chunks[1] - 1 and (
                    chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1
                ):
                    """ If chunk is on the bottom egde but not corners"""
                    prediction_output = prediction_output[
                        :,
                        :chunk_size,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                    ]
                    """ bottom egde window"""
                    final_window = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                elif chunk_location[1] == num_chunks[1] - 1 and chunk_location[2] == 0:
                    """ If chunk is on the bottom left corner"""
                    prediction_output = prediction_output[
                        :,
                        :chunk_size,
                        :chunk_size,
                    ]
                    """ bottom left window"""
                    window_l = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_l[:step, :step], window_l[:step, step:]],
                            [np.ones((step, step)), window_b[step:, step:]],
                        ]
                    )
                elif chunk_location[1] == 0 and chunk_location[2] == 0:
                    """ If chunk is on the top left corner"""
                    prediction_output = prediction_output[:, :chunk_size, :chunk_size]
                    """ Top left window"""
                    window_u = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
                    window_l = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
                    final_window = np.block(
                        [
                            [np.ones((step, step)), window_u[:step, step:]],
                            [window_l[step:, :step], window_l[step:, step:]],
                        ]
                    )
                elif chunk_location[2] == 0 and (
                    chunk_location[1] > 0 and chunk_location[1] < num_chunks[1]
                ):
                    """ If chunk is on the left edge but not corners"""
                    prediction_output = prediction_output[
                        :,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                        :chunk_size,
                    ]
                    """ top edge window"""
                    final_window = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
                elif (
                    chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1
                ) and (chunk_location[1] == 0):
                    """ If chunk is on the top edge but not corners"""
                    prediction_output = prediction_output[
                        :,
                        :chunk_size,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                    ]
                    """ top edge window"""
                    final_window = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
                elif (
                    chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
                ) and (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1):
                    """ if the chunk is in the middle"""
                    prediction_output = prediction_output[
                        :,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                    ]
                    final_window = window

                """ 
                slices = dask.array.core.slices_from_chunks(
                    dask.array.empty(chunk_data_.shape).chunks
                )  #Create slices for iterating over and running the model
                prediction_output = np.empty(
                    shape=(5, chunk_data_.shape[1], chunk_data_.shape[2])
                )  #Create the prediction_outputput but empty
                for slice_ in slices:
                    tensor = torch.as_tensor(chunk_data_[slice_][np.newaxis, ...]).to(
                        device
                    )
                    with torch.no_grad():
                        prediction_output[:, slice_[1], slice_[2]] = model(tensor).cpu().numpy()[0]
                    del tensor  # Delete the tensor to free up memory
                """

                if prediction_output.shape[1:] != final_window.shape:
                    logging.error(
                        f" In runModel the shape of window and the array do not match"
                        f" The shape of window is {final_window.shape}"
                        f" The shape of array is {prediction_output.shape[1:]}"
                    )
                if prediction_output.shape[1:] != (chunk_size, chunk_size):
                    logging.error(
                        f" In runModel the shape of the array is not the expected shape"
                        f" The shape of array is {prediction_output.shape[1:]}"
                    )
                else:
                    return np.concatenate(
                        (
                            prediction_output * final_window,
                            final_window[np.newaxis, :, :],
                        ),
                        axis=0,
                    )
            except Exception as e:
                logging.error(f"Error occured in IrunModel: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Release unused memory

    @staticmethod
    def runModel_partial_neighbor(
        chunk_data: np.ndarray,
        chunk_size: int,
        model_path: str,
        block_info=None,
    ):
        num_chunks = block_info[0]["num-chunks"]
        chunk_location = block_info[0]["chunk-location"]
        if chunk_data.size > 0 and chunk_data is not None:
            try:
                # Defining the base window for window creation later
                step = chunk_size >> 1
                window = w.hann(M=chunk_size, sym=False)
                window = window[:, np.newaxis] * window[np.newaxis, :]
                final_window = np.empty((1, 1))
                chunk_data_ = chunk_data

                # Getting a (chunk_size, chunk_size) chunk with its window
                if chunk_location[2] == num_chunks[2] - 1 and chunk_location[1] == 0:
                    """ If chunk is top right corner"""
                    chunk_data_ = chunk_data_[
                        :,
                        :chunk_size,
                        :chunk_size,
                    ]
                    """ Top right corner window"""
                    window_u = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_u[:step, :step], np.ones((step, step))],
                            [window_r[step:, :step], window_r[step:, step:]],
                        ]
                    )
                elif chunk_location[2] == num_chunks[2] - 1 and (
                    chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
                ):
                    """ If chunk is on the right egde but not corners"""
                    chunk_data_ = chunk_data[
                        :,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                        :chunk_size,
                    ]
                    """ left egde window """
                    final_window = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                elif chunk_location[2] == num_chunks[2] - 1 and (
                    chunk_location[1] == num_chunks[1] - 1
                ):
                    """ If chunk is in the bottom right corner"""
                    chunk_data_ = chunk_data_[
                        :,
                        :chunk_size,
                        :chunk_size,
                    ]
                    """ bottom right window """
                    window_r = np.hstack(
                        [
                            window[:, :step],
                            np.tile(window[:, step : step + 1], (1, step)),
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_r[:step, :step], window_r[:step, step:]],
                            [window_b[step:, :step], np.ones((step, step))],
                        ]
                    )
                elif chunk_location[1] == num_chunks[1] - 1 and (
                    chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1
                ):
                    """ If chunk is on the bottom egde but not corners"""
                    chunk_data_ = chunk_data_[
                        :,
                        :chunk_size,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                    ]
                    """ bottom egde window"""
                    final_window = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                elif chunk_location[1] == num_chunks[1] - 1 and chunk_location[2] == 0:
                    """ If chunk is on the bottom left corner"""
                    chunk_data_ = chunk_data_[
                        :,
                        :chunk_size,
                        :chunk_size,
                    ]
                    """ bottom left window"""
                    window_l = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
                    window_b = np.vstack(
                        [
                            window[:step, :],
                            np.tile(window[step : step + 1, :], (step, 1)),
                        ]
                    )
                    final_window = np.block(
                        [
                            [window_l[:step, :step], window_l[:step, step:]],
                            [np.ones((step, step)), window_b[step:, step:]],
                        ]
                    )
                elif chunk_location[1] == 0 and chunk_location[2] == 0:
                    """ If chunk is on the top left corner"""
                    chunk_data_ = chunk_data[:, :chunk_size, :chunk_size]
                    """ Top left window"""
                    window_u = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
                    window_l = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
                    final_window = np.block(
                        [
                            [np.ones((step, step)), window_u[:step, step:]],
                            [window_l[step:, :step], window_l[step:, step:]],
                        ]
                    )
                elif chunk_location[2] == 0 and (
                    chunk_location[1] > 0 and chunk_location[1] < num_chunks[1]
                ):
                    """ If chunk is on the left edge but not corners"""
                    chunk_data_ = chunk_data[
                        :,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                        :chunk_size,
                    ]
                    """ top edge window"""
                    final_window = np.hstack(
                        [
                            np.tile(window[:, step : step + 1], (1, step)),
                            window[:, step:],
                        ]
                    )
                elif (
                    chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1
                ) and (chunk_location[1] == 0):
                    """ If chunk is on the top edge but not corners"""
                    chunk_data_ = chunk_data[
                        :,
                        :chunk_size,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                    ]
                    """ top edge window"""
                    final_window = np.vstack(
                        [
                            np.tile(window[step : step + 1, :], (step, 1)),
                            window[step:, :],
                        ]
                    )
                elif (
                    chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
                ) and (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1):
                    """ if the chunk is in the middle"""
                    chunk_data_ = chunk_data[
                        :,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                        int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                    ]
                    final_window = window

                device = torch.device("cpu")
                device_id = None
                if torch.cuda.is_available():
                    num_devices = torch.cuda.device_count()
                    for i in range(num_devices):
                        res = {"gpu": torch.cuda.utilization(i)}
                        torch_cuda_mem = torch.cuda.mem_get_info(i)
                        mem = {
                            "used": torch_cuda_mem[-1] - torch_cuda_mem[0],
                            "total": torch_cuda_mem[-1],
                        }
                        used_ram = mem["used"] / (1024**2)
                        max_ram = mem["total"] / (1024**2)
                        used_ram_percentage = (used_ram / max_ram) * 100
                        if used_ram_percentage < 70 and res["gpu"] < 70:
                            device_id = i
                            break

                """ 
                device = (
                    torch.device(f"cuda:{device_id}")
                    if device_id is not None
                    else torch.device("cpu")
                )
                """
                device = torch.device("cuda:0")
                """
                if (
                    chunk_location[2] > int(num_chunks[2] / num_devices)
                    and chunk_location[2] <= int(num_chunks[2] / num_devices) * 2
                ):
                    device = torch.device("cuda:1")
                elif chunk_location[2] > int(num_chunks[2] / num_devices) * 2:
                    device = torch.device("cuda:2")
                """
                model = torch.jit.load(model_path, map_location=device).to(
                    device
                )  # load the model
                # convert chunck to torch Tensor
                tensor = torch.as_tensor(chunk_data_[np.newaxis, ...]).to(model.device)

                out = np.empty(
                    shape=(5, chunk_data_.shape[1], chunk_data_.shape[2])
                )  # Create the output but empty
                # run the model
                with torch.no_grad():
                    out = model(tensor).cpu().numpy()[0]
                """ 
                slices = dask.array.core.slices_from_chunks(
                    dask.array.empty(chunk_data_.shape).chunks
                )  #Create slices for iterating over and running the model
                out = np.empty(
                    shape=(5, chunk_data_.shape[1], chunk_data_.shape[2])
                )  #Create the output but empty
                for slice_ in slices:
                    tensor = torch.as_tensor(chunk_data_[slice_][np.newaxis, ...]).to(
                        device
                    )
                    with torch.no_grad():
                        out[:, slice_[1], slice_[2]] = model(tensor).cpu().numpy()[0]
                    del tensor  # Delete the tensor to free up memory
                """
                del tensor
                del model  # Delete the model if it's not needed anymore
                if out.shape[1:] != final_window.shape:
                    logging.error(
                        f" In runModel the shape of window and the array do not match"
                        f" The shape of window is {final_window.shape}"
                        f" The shape of array is {out.shape[1:]}"
                    )
                if out.shape[1:] != (chunk_size, chunk_size):
                    logging.error(
                        f" In runModel the shape of the array is not the expected shape"
                        f" The shape of array is {out.shape[1:]}"
                    )
                else:
                    return np.concatenate(
                        (out * final_window, final_window[np.newaxis, :, :]), axis=0
                    )
                """
                # convert chunck to torch Tensor
                tensor = torch.tensor(chunk_data[None, :, :, :]).to(device)
                # run the model
                with torch.no_grad():
                    model_tensor = model(tensor).argmax(dim=1).to(torch.uint8)
                return model_tensor.cpu().numpy()[0]

                
                if model_tensor.shape[1] == 1:
                    features = expit(model_tensor).cpu().numpy()[0]
                    return (
                        np.where(features > 0.5, 1, 0)
                        .squeeze(0)
                        .astype(np.uint8)
                    )
                else:
                    features = model_tensor.to(torch.uint8)
                    return features.cpu().numpy()[0]
                    """
            except Exception as e:
                logging.error(f"Error occured in IrunModel: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Release unused memory

    def write_inference_to_tif(
        self,
        mask_image: np.ndarray,
    ):
        mask_image = mask_image[
            np.newaxis, : mask_image.shape[0], : mask_image.shape[1]
        ]
        self.raster_meta.update(
            {
                "driver": "GTiff",
                "height": mask_image.shape[1],
                "width": mask_image.shape[2],
                "count": mask_image.shape[0],
                "dtype": "uint8",
                "compress": "lzw",
            }
        )
        with rasterio.open(
            "/space/partner/nrcan/geobase/work/dev/datacube/parallel/Change_detection_Results/geo_deep_learning.tif",
            "w+",
            **self.raster_meta,
        ) as dest:
            dest.write(mask_image)


def aois_from_csv(
    csv_path: Union[str, Path],
    bands_requested: List = [],
    attr_field_filter: str = None,
    attr_values_filter: str = None,
    data_dir: str = "data",
    for_multiprocessing=False,
    raster_stats=False,
    equalize_clahe_clip_limit: int = 0,
    chunk_size: int = 512,
):
    """
    Creates list of AOIs by parsing a csv file referencing input data
    @param csv_path:
        path to csv file containing list of input data. See README for details on expected structure of csv.
    N.B.: See AOI docstring for information on other parameters.
    Returns: a list of AOIs objects
    """
    aois = []
    data_list = read_csv(csv_path)
    logging.info(
        f"\n\tSuccessfully read csv file: {Path(csv_path).name}\n"
        f"\tNumber of rows: {len(data_list)}\n"
        f"\tCopying first row:\n{data_list[0]}\n"
    )
    with tqdm(
        enumerate(data_list), desc="Creating AOI's", total=len(data_list)
    ) as _tqdm:
        for i, aoi_dict in _tqdm:
            _tqdm.set_postfix_str(f"Image: {Path(aoi_dict['tif']).stem}")
            try:
                new_aoi = AOI.from_dict(
                    aoi_dict=aoi_dict,
                    bands_requested=bands_requested,
                    attr_field_filter=attr_field_filter,
                    attr_values_filter=attr_values_filter,
                    root_dir=data_dir,
                    for_multiprocessing=for_multiprocessing,
                    equalize_clahe_clip_limit=equalize_clahe_clip_limit,
                    raster_stats=raster_stats,
                    chunk_size=chunk_size,
                )
                logging.debug(new_aoi)
                aois.append(new_aoi)
            except FileNotFoundError as e:
                logging.error(
                    f"{e}\nGround truth file may not exist or is empty.\n"
                    f"Failed to create AOI:\n{aoi_dict}\n"
                    f"Index: {i}"
                )
    return aois


def get_tiff_paths_from_csv(
    csv_path: Union[str, Path],
):
    """
    Creates list of AOI dict by parsing a csv file referencing input data
    @param csv_path:
        path to csv file containing list of input data. See README for details on expected structure of csv.
    Returns: A list of tiff path
    """
    aois_dictionary = []
    data_list = read_csv(csv_path)
    logging.info(
        f"\n\tSuccessfully read csv file: {Path(csv_path).name}\n"
        f"\tNumber of rows: {len(data_list)}\n"
        f"\tCopying first row:\n{data_list[0]}\n"
    )
    with tqdm(
        enumerate(data_list), desc="Creating A list of tiff paths", total=len(data_list)
    ) as _tqdm:
        for i, aoi_dict in _tqdm:
            _tqdm.set_postfix_str(f"Image: {Path(aoi_dict['tif']).stem}")
            try:
                aois_dictionary.append(aoi_dict)
            except FileNotFoundError as e:
                logging.error(
                    f"{e}" f"Failed to get the path of :\n{aoi_dict}\n" f"Index: {i}"
                )
    return aois_dictionary


def single_aoi(
    aoi_dict: dict,
    bands_requested: List = [],
    attr_field_filter: str = None,
    attr_values_filter: str = None,
    data_dir: str = "data",
    for_multiprocessing=False,
    raster_stats=False,
    equalize_clahe_clip_limit: int = 0,
    chunk_size: int = 512,
):
    """
    Creates a single AOI from the provided tiff path of the csv file referencing input data
    @param tiff_path:
        path to tiff file containing data.
    Returns: a single AOU object
    """
    try:
        new_aoi = AOI.from_dict(
            aoi_dict=aoi_dict,
            bands_requested=bands_requested,
            attr_field_filter=attr_field_filter,
            attr_values_filter=attr_values_filter,
            root_dir=data_dir,
            for_multiprocessing=for_multiprocessing,
            equalize_clahe_clip_limit=equalize_clahe_clip_limit,
            raster_stats=raster_stats,
            chunk_size=chunk_size,
        )
        logging.debug(new_aoi)
    except FileNotFoundError as e:
        logging.error(
            f"{e}\nGround truth file may not exist or is empty.\n"
            f"Failed to create AOI:\n{aoi_dict}\n"
        )
    return new_aoi
