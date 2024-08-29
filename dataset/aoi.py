import gc
import os
import sys
import json
import dask
import torch
import pyproj
import pystac
import logging
import requests
import rasterio
import functools
import threading
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import dask.array as da
import geopandas as gpd
from pathlib import Path
from skimage import exposure
from pandas.io.common import is_url
from hydra.utils import to_absolute_path
from kornia.enhance import equalize_clahe
from multiprocessing.pool import ThreadPool
from omegaconf import listconfig, ListConfig
from kornia import image_to_tensor, tensor_to_image
from typing import Union, Sequence, Tuple, List, Dict
from dask.diagnostics import ResourceProfiler, ProgressBar
from rasterio.plot import reshape_as_image, reshape_as_raster

if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))


from utils.logger import get_logger
from dataset.stacitem import SingleBandItemEO
from utils.utils import read_csv, minmax_scale, download_url_wcheck
from utils.verifications import (
    assert_crs_match,
    validate_raster,
    validate_features_from_gpkg,
    validate_num_bands,
)
from utils.geoutils import (
    stack_singlebands_vrt,
    is_stac_item,
    subset_multiband_vrt,
    check_rasterio_im_load,
    check_gdf_load,
    bounds_gdf,
    bounds_riodataset,
    overlap_poly1_rto_poly2,
    gdf_mean_vertices_nb,
    xarray_profile_attrs,
)

logging = get_logger(__name__)

class AOI(object):
    """
    Object containing all data information about a single area of interest
    based on https://github.com/stac-extensions/ml-aoi
    Note: it only saves to tiff file 
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
        write_dest_raster: bool = False,
        write_dest_zarr: bool = False,
        raster_stats: bool = False,
        equalize_clahe_clip_limit: int = 0,
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
        @param write_dest_raster: bool, optional
            If True, a multi-band raster side by side with single-bands rasters as provided in input csv.
        @param write_dest_zarr: bool, optional
            If True, a zarr file along json metadata will be saved.    
        @param raster_stats:
            if True, radiometric stats will be read from Stac Item if available or calculated
        @param equalize_clahe_clip_limit: int, optional
            Threshold value for contrast limiting. If 0 clipping is disabled.
            Geo-deep-learning enforces the use of an integer to avoid confusion with sklearn's CLAHE algorithm, which
            expects float between 0 and 1. See:
            https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
            https://kornia.readthedocs.io/en/latest/enhance.html#kornia.enhance.equalize_clahe
        """
        self.raster_np = None
        self.raster_name = Path(raster)  # default name, may be overwritten later
        self.label = None
        self.label_gdf = None
        self.label_invalid_features = None
        self.label_bounds = None
        self.overlap_label_rto_raster = self.overlap_raster_rto_label = None
        self.epsg_raster = self.epsg_label = None
        self.crs_match = None

        """ ---------------------------Input Validation-------------------------------"""
        if not isinstance(raster, (str, Path)):
            raise TypeError(
                f"Raster path should be a string or a Path object.\nGot {raster} of type {type(raster)}"
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

        if not isinstance(equalize_clahe_clip_limit, int):
            raise ValueError(
                f"Enhance clip limit should be an integer. See documentation.\n"
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
            )
            split = "inference"
        self.split = split

        if aoi_id and not isinstance(aoi_id, str):
            raise TypeError(
                f"AOI name should be a string. Got {aoi_id} of type {type(aoi_id)}"
            )

        # Check collection string
        if collection and not isinstance(collection, str):
            raise TypeError(
                f"Collection name should be a string. Got {collection} of type {type(collection)}"
            )
        self.aoi_id = aoi_id

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
        self.write_dest_zarr = write_dest_zarr
        
        if not isinstance(for_multiprocessing, bool):
            raise ValueError(
                f'\n"for_multiprocessing" should be a boolean.\nGot {for_multiprocessing}.'
            )
        self.for_multiprocessing = for_multiprocessing
        self.root_dir = Path(root_dir)
        
        rio_gdal_options = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        }
        if self.for_multiprocessing:
            dask.config.set(scheduler='threads', num_workers=8)
            dask.config.set(pool=ThreadPool(8))
        else:
            dask.config.set(scheduler='threads')

        """ -------------------------------------------------------------------"""

        """ -------------------------Processing---------------------------------"""
        # it constructs a list of URLs or file paths for raster bands: This function now returns a dict of {band:url}
        self.raster_parsed = self.parse_input_raster(
            csv_raster_str=self.raster_raw_input,
            raster_bands_requested=self.raster_bands_request,
        )
        logging.info(f"Successfully parsed Rasters: \n {self.raster_parsed}\n")

        # Chack the size of tiff files
        for single_raster in self.raster_parsed.values():
            size = (
                Path(single_raster).stat().st_size
                if not is_url(single_raster)
                else self.get_tiff_size(single_raster)
            )
            logging.debug(
                f"Raster to validate: {raster}\n"
                f"Size: {size}\n"
                f"Extended check: {False}"
            )
        
        # If stac item input, keep Stac item object as attribute
        if is_stac_item(self.raster_raw_input):
            item = SingleBandItemEO(
                item=pystac.Item.from_file(self.raster_raw_input),
                bands_requested=self.raster_bands_request,
            )
            self.raster_stac_item = item
            self.raster_stats = self.read_stack_stat() if raster_stats else {} 
        else:
            self.raster_stac_item = None
            self.raster_stats = {}
        
        for band, url in self.raster_parsed.items():
            if is_url(url):
                self.url_was_provided = True
                out_name = self.root_dir / Path(url).name
                download_url_wcheck(url, root=str(out_name.parent), filename=out_name.name)
                self.raster_parsed[band] = str(out_name)
            else:
                break
        
        # check if the image needs contrast enhancement
        decimated_image=[]
        with rasterio.Env(**rio_gdal_options):
            for band in self.raster_parsed.values():
                with rasterio.open(band, 'r') as src:
                    overview_factors = src.overviews(1)
                    second_highest_factor = 1
                    if len(overview_factors) > 2: 
                        second_highest_factor = overview_factors[-2]
                    decimated_image.append(src.read(1, out_shape=(1, int(src.height / second_highest_factor), int(src.width / second_highest_factor))))
        self.high_or_low_contrast = exposure.is_low_contrast(np.mean(np.asarray(decimated_image), axis=0).astype(np.uint8), 0.3)
        del decimated_image
        
        self.raster_src_is_multiband = self.raster_needs_vrt = False
        self.raster = rasterio.open(next(iter(self.raster_parsed.values())))
        print(self.raster)
        self.raster_stats = (
            self.calc_raster_stats()
            if raster_stats
            else None
        )

        if len(self.raster_parsed) == 1:
            import rioxarray
            self.aoi_xr_array= rioxarray.open_rasterio(list(self.raster_parsed.values())[0], 
                                                        chunks=(1, min(int(self.raster.height / 4), 15000), min(int(self.raster.width / 4), 15000)))
            if len(self.raster_bands_request) != self.raster.count:
                self.raster_needs_vrt = True
                self.aoi_xr_array = self.aoi_xr_array[
                    [idx - 1 for idx in self.raster_bands_request], :, :
                ]
            else:
                self.raster_needs_vrt = False
                self.raster_bands_request = None
        else:  # If parsed result has more than a single file, then we're dealing with single-band files
            self.raster_needs_vrt = True
            all_bands_requested = []
            # the rioxarray should be imported any where we need it
            with rasterio.Env(**rio_gdal_options):
                import rioxarray
                for band, url in self.raster_parsed.items():
                    all_bands_requested.append(rioxarray.open_rasterio(url,
                                                                       chunks=(1, min(int(self.raster.height / 4), 15000), min(int(self.raster.width / 4), 15000))))
            self.aoi_xr_array = xr.concat(all_bands_requested, dim="band")
        

        self.raster_name = self.name_raster(
            input_path=self.raster_raw_input,
            bands_list=self.raster_bands_request,
            root_dir=self.root_dir,
        )
        if aoi_id is None:
            self.aoi_id = self.raster_name.stem 
        
        self.raster_dest, self.raster_is_vrt = self.raster_src_to_dest()
        self.raster = check_rasterio_im_load(self.raster_dest)
        validate_raster(self.raster)
        self.raster_meta = self.raster.meta
        self.raster_bounds = bounds_riodataset(self.raster)
        
        # Check label data
        if label:
            
            self.label = Path(label)
            self.label_gdf = check_gdf_load(label)
            self.label_invalid_features = validate_features_from_gpkg(label)

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
                self.crs_match, self.epsg_raster, self.epsg_label = assert_crs_match(
                    self.raster, self.label_gdf
                )
            except pyproj.exceptions.CRSError as e:
                logging.warning(
                    f"\nError while checking CRS match between raster and label."
                    f"\n{e}"
                )
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

        if self.enhance_clip_limit > 0 and self.high_or_low_contrast:   
            self.aoi_xr_array_dims = self.aoi_xr_array.dims
            self.aoi_xr_array= da.map_overlap(
                self.equalize_adapthist_enhancement,
                self.aoi_xr_array.data,
                per_band=True,
                depth={
                    1: min(int(self.aoi_xr_array.shape[1] / 20), 4000),
                    2: min(int(self.aoi_xr_array.shape[2] / 20), 4000),
                },
                trim=True,
                boundary="none",
                dtype=self.aoi_xr_array.dtype,
            )
            if not write_dest_zarr and write_dest_raster:
                self.aoi_xr_array = xr.DataArray(self.aoi_xr_array, dims=self.aoi_xr_array_dims, attrs=xarray_profile_attrs(input_raster=self.raster))
                self.aoi_xr_array.name ="Enhanced"
                self.raster_name = (
                    self.raster_name.parent
                    / f"{self.raster_name.stem}_clahe{self.enhance_clip_limit}.tif"
                )
                self.aoi_id = self.raster_name.stem
            elif write_dest_zarr and not write_dest_raster:
                self.raster_name = (
                    self.raster_name.parent
                    / f"{self.raster_name.stem}_clahe{self.enhance_clip_limit}.zarr"
                )
                self.aoi_id = self.raster_name.stem
        if write_dest_zarr and not write_dest_raster:
            self.raster_json = (
                self.raster_name.parent /
                f"{self.raster_name.stem}.json"
            )
        
        with ProgressBar() as pbar:
            pbar.register()
            gc.collect()
            if write_dest_raster and not self.write_dest_zarr:
                import rioxarray
                self.aoi_xr_array.rio.to_raster(
                    to_absolute_path(str(self.raster_name)), tiled=True, lock=threading.Lock())
                logging.info(f"Wrote destination Raster to : {self.raster_name}")
                self.raster = check_rasterio_im_load(self.raster_name)
                del self.aoi_xr_array
            elif self.write_dest_zarr and not write_dest_raster:
                self.aoi_xr_array.rechunk((self.raster.count, 512,512)).to_zarr(to_absolute_path(str(self.raster_name)), overwrite=True)
                serializable_attrs = {key: self._serializable_json(value) for key, value in xarray_profile_attrs(input_raster=self.raster).items()}
                with open(self.raster_json, 'w', encoding='utf-8') as json_file:
                    json.dump(serializable_attrs, json_file, indent=4)
                logging.info(f"Wrote destination Zarr to : {self.raster_name}")
                logging.info(f"Wrote destination Metadata to : {self.raster_json}")
                del self.aoi_xr_array, serializable_attrs
        
        if (
            raster_num_bands_expected and not self.raster_is_vrt
        ):  # VRT necessarily contains expected number of bands
            validate_num_bands(
                raster_path=self.raster, num_bands=raster_num_bands_expected
            )

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
        
        # removing all raw data that was downloaded
        if self.raster_stac_item is not None:
            for band, path in self.raster_parsed.items():
                if os.path.isfile(to_absolute_path(path)):
                    os.remove(to_absolute_path(path))
        gc.collect()
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
        write_dest_raster: bool = False,
        write_dest_zarr: bool = False,
        equalize_clahe_clip_limit: int = 0,
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
                f" Only imagery will be processed; No ground truth data found for {aoi_dict['tif']}.\n"
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
            root_dir=root_dir,
            for_multiprocessing=for_multiprocessing,
            raster_stats=raster_stats,
            equalize_clahe_clip_limit=equalize_clahe_clip_limit,
            write_dest_raster=write_dest_raster,
            write_dest_zarr=write_dest_zarr,
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

    def raster_read(self):
        self.raster_np = (
            self.raster.read() if self.raster_np is None else self.raster_np
        )
        return self.raster_np
    
    def raster_src_to_dest(self):
        """
        Outputs a processing-ready raster after merge from singleband source rasters or after reordering or selecting
        subset of bands from a source multiband raster.
        """

        self.raster_is_vrt = False
        if self.raster_needs_vrt:
            self.raster_src_is_multiband = False
            if self.raster_src_is_multiband:
                if not all(
                    [isinstance(band, int) for band in self.raster_bands_request]
                ):
                    raise ValueError(
                        f"Use only a list of integers to select bands from a multiband raster.\n"
                        f"Got {self.raster_bands_request}"
                    )
                # TODO: open the raw raster only once when initialize the AOI. Otherwise, we open it all the time here,
                #       thus, slowing down the tiling process:
                if (
                    len(self.raster_bands_request)
                    > rasterio.open(self.raster_raw_input).count
                ):
                    raise ValueError(
                        f"Trying to subset more bands than actual number in source raster.\n"
                        f"Requested: {self.raster_bands_request}\n"
                        f"Available: {rasterio.open(self.raster_raw_input).count}"
                    )
                self.raster_dest = subset_multiband_vrt(
                    list(self.raster_parsed.values())[0],
                    band_request=self.raster_bands_request,
                )
            else:
                self.raster_dest = stack_singlebands_vrt(
                    list(self.raster_parsed.values())
                )
            self.raster_is_vrt = True
        elif self.raster_bands_request:
            raise ValueError(
                "Cannot select or reorder with requested bands. Make sure your source raster(s) are "
                "in expected formats. See README."
            )
        else:  # source raster can be used as is
            self.raster_dest = str(list(self.raster_parsed.values())[0])

        return self.raster_dest, self.raster_is_vrt
    
    def _serializable_json(self, value):
        """Convert non-serializable values to serializable ones."""
        if isinstance(value, (str, int, float, list, dict)):
            return value
        if isinstance(value, dict):
            return {k: self._serializable_json(v) for k, v in value.items()}
        return str(value)  # Convert other non-serializable types to string
    
    def get_tiff_size(self, url):
        """This funcstion is for getting the size of tiffs that have been provided by urls"""
        tiff_info = requests.head(url)
        if tiff_info.status_code == 200:
            tiff_length = tiff_info.headers.get("Content-Length")
            return int(tiff_length) if tiff_length else None
        else:
            raise Exception(
                f"Failed to retrieve headers, status code: {tiff_info.status_code}"
            )

    def name_raster(
        self,
        input_path: Union[str, Path],
        bands_list: Sequence = None,
        root_dir: Union[str, Path] = "data",
    ):
        """
        Assigns a name to the AOI's raster considering different input types.
        Used for logging and as output name for .tif built from a VRT
        (see self.write_multiband_from_singleband_rasters_as_vrt())
        @param root_dir: Root directory where derived raster would be written.
        @param input_path: path to input raster file as accepted by GDL (see dataset/README.md)
        @param bands_list: list of requested bands from input raster
        """
        if not bands_list:  # multiband, no band selection or ordering
            return Path(input_path)
        raster_name_parent = Path(root_dir)

        bands_list_str = [str(band) for band in bands_list]
        if len(bands_list_str) <= 4:
            bands_suffix = f"{'-'.join(bands_list_str)}"
        else:  # e.g. hyperspectral imagery
            bands_suffix = f"{len(bands_list)}bands"

        if (
            "${dataset.bands}" in input_path
        ):  # singleband with ${dataset.bands} pattern (implies band selection)
            if not self.write_dest_zarr:
                raster_name = (
                    raster_name_parent
                    / f"{Path(input_path).stem.replace('${dataset.bands}', bands_suffix)}.tif"
                )
            else:
                raster_name = (
                    raster_name_parent
                    / f"{Path(input_path).stem.replace('${dataset.bands}', bands_suffix)}.zarr"
                )
        elif (
            len(bands_list_str) > 0
        ):  # singleband from stac item or multiband with band selection
            if not self.write_dest_zarr:
                raster_name = (
                    raster_name_parent / f"{Path(input_path).stem}_{bands_suffix}.tif"
                )
            else:
                raster_name = (
                    raster_name_parent / f"{Path(input_path).stem}_{bands_suffix}.zarr"
                )
        else:
            raise ValueError(
                "Invalid input raster. See README for valid input raster formats"
            )
        return raster_name

    def read_stack_stat(self):
        """For stac items formatted as expected, reads mean and std of raster imagery, per band.
        See stac item example: https://datacube-stage.services.geo.ca/api/collections/worldview-4-ortho-pansharp/items/Sherbrooke2018-013827026010_01_P001-WV04"""

        if self.raster_bands_request:
            stats = {name: {} for name in self.raster_bands_request}
        else:
            stats = {f"band_{index}": {} for index in range(self.raster.count)}
        try:
            stats_asset = self.raster_stac_item.item.assets["STATS"]
            if is_url(stats_asset.href):
                response = requests.get(stats_asset.href)
                response.raise_for_status()  # Ensure we raise an error for bad responses
                stac_stats = response.json()  # Directly parse the JSON response
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
                download_url_wcheck(stats_asset.href, root=str(self.root_dir), filename=Path(stats_asset.href).name)
                stats_href = self.root_dir / Path(stats_asset.href).name
            else:
                stats_href = to_absolute_path(stats_asset.href)
            with open(stats_href, 'r') as ifile:
                stac_stats = json.loads(ifile.read())
            stac_stats = {bandwise_stats['asset']: bandwise_stats for bandwise_stats in stac_stats}
            for band in self.raster_stac_item.bands:
                stats[band.common_name] = stac_stats[band.name]
        except (AttributeError, KeyError):
            self.raster_np = self.raster_read()
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
                           "median": mean_median, "std": mean_std, "low_contrast": self.high_or_low_contrast},
            "histogram": {"buckets": mean_hist}}
        return stats
    
    def equalize_adapthist_enhancement(
        self,
        aoi_chunk: np.ndarray,
        per_band: bool = True,
    ):
        
        """This function applies equalize_adapthist on each chunk of dask array
        each chunk is [band, height, width] --> So, the contrast enhancement is applied on each hand separately"""

        if aoi_chunk.size > 0 and aoi_chunk is not None:
            if per_band:
                ready_np_array = image_to_tensor(reshape_as_image(aoi_chunk))
                ready_np_array = minmax_scale(
                    img=ready_np_array, scale_range=(0, 1), orig_range=(0, 255)
                )
                ready_np_array = equalize_clahe(
                    ready_np_array.float().unsqueeze(0),
                    clip_limit=float(self.enhance_clip_limit),
                    grid_size=(3, 3),
                )
                ready_np_array = tensor_to_image(
                    (ready_np_array * 255).long(), keepdim=True
                ).squeeze()
                torch.cuda.empty_cache()
                return ready_np_array[np.newaxis, ...].astype(np.uint8)
            else:
                ready_np_array = image_to_tensor(reshape_as_image(aoi_chunk))
                ready_np_array = minmax_scale(
                    img=ready_np_array, scale_range=(0, 1), orig_range=(0, 255)
                )
                ready_np_array = equalize_clahe(
                    ready_np_array.float().unsqueeze(0),
                    clip_limit=float(self.enhance_clip_limit),
                    grid_size=(3, 3),
                )
                ready_np_array = tensor_to_image((ready_np_array * 255).long())
                ready_np_array = reshape_as_raster(ready_np_array)
                torch.cuda.empty_cache()
                return ready_np_array
        else:
            return aoi_chunk

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
                validate_raster(rasterio.open(csv_raster_str))
                raster["all_bands"] = csv_raster_str
                return raster
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


def aois_from_csv(
    csv_path: Union[str, Path],
    bands_requested: List = [],
    attr_field_filter: str = None,
    attr_values_filter: str = None,
    data_dir: str = "data",
    for_multiprocessing=False,
    raster_stats=False,
    equalize_clahe_clip_limit: int = 0,
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