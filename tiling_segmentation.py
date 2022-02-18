import csv
import difflib
import functools
import glob
import math
import multiprocessing

import matplotlib.pyplot
import omegaconf.listconfig
import os
from typing import Union, Sequence
from pathlib import Path

import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box
from tqdm import tqdm
from omegaconf import DictConfig, open_dict

# Our modules
from utils.utils import (
    get_key_def, read_csv, get_git_hash, select_modalities
)
from utils.verifications import (
    validate_raster, assert_crs_match, validate_num_bands
)
# Set the logging file
from utils import utils
logging = utils.get_logger(__name__)  # import logging
# Set random seed for reproducibility
np.random.seed(1234)

from solaris_gdl import tile
from solaris_gdl import vector
from solaris_gdl.utils.core import _check_rasterio_im_load, _check_gdf_load
from solaris_gdl.utils.geo import reproject_geometry


class AOI(object):
    """
    Object containing all data information about a single area of interest
    """

    def __init__(self, img: Union[Path, str],
                 dataset: str,
                 gt: Union[Path, str] = None,
                 attr_field: str = None,
                 attr_vals: Sequence = None,
                 name: str = None,
                 index: str = None,
                 tiles_dir: Union[Path, str] = None):
        """

        @param img: pathlib.Path or str
            Path to source imagery
        @param dataset: str
            Name of destination dataset for aoi. Should be 'trn', 'tst' or 'inference'
        @param gt: pathlib.Path or str
            Path to ground truth file. If not provided, AOI is considered only for inference purposes
        @param attr_field: str, optional
            Name of attribute field used to filter features. If not provided all geometries in ground truth file
            will be considered.
        @param attr_vals: list of ints, optional
            The list of attribute values in given attribute field used to filter features from ground truth file.
            If not provided, all values in field will be considered
        @param name: str
            Name of area of interest. Used to name output folders. Multiple AOI instances can bear the same name.
        @param index: int
            Index to assign to aoi. Useful when creating multiple aois at once (from for loop, for example)
        @param tiles_dir: Path or str
            Path where output tiles will be written
        """
        if not isinstance(img, (Path, str)):
            raise TypeError(f'Image path should be a of class pathlib.Path or a string.\n'
                            f'Got {img} of type {type(img)}')
        if not Path(img).is_file():
            raise FileNotFoundError(f'{img} is not a valid file')
        self.img = Path(img)

        if gt and not isinstance(gt, (Path, str)):
            raise TypeError(f'Ground truth path should be a of class pathlib.Path or a string.\n'
                            f'Got {gt} of type {type(gt)}')
        elif gt and (not Path(gt).is_file() or os.stat(gt).st_size == 0):
            raise FileNotFoundError(f'{gt} is not a valid file')
        elif gt:
            self.gt = Path(gt)
            # creating overhead and has caused pyproj.exceptions.CRSError. Try/except needed minimally.
            # self.crs_match, self.epsg_raster, self.epsg_gt = assert_crs_match(self.img, self.gt)
        else:
            self.gt = gt
            # self.crs_match = self.epsg_raster = self.epsg_gt = None

        # Defaults to image metadata as given by rasterio
        img_meta = _check_rasterio_im_load(self.img).meta
        self.img_meta = img_meta

        if gt and attr_field and not isinstance(attr_field, str):
            raise TypeError(f'Attribute field name should be a string.\n'
                            f'Got {attr_field} of type {type(attr_field)}')
        self.attr_field = attr_field

        if gt and attr_vals and not isinstance(attr_vals, (list, omegaconf.listconfig.ListConfig)):
            raise TypeError(f'Attribute values should be a list.\n'
                            f'Got {attr_vals} of type {type(attr_vals)}')
        self.attr_vals = attr_vals

        # FIXME: if no ground truth is inputted, dataset should be optional.
        if not isinstance(dataset, str) and dataset not in ['trn', 'tst', 'inference']:
            raise ValueError(f"Dataset should be a string: 'trn', 'tst' or 'inference'. Got {dataset}.")
        elif not gt and dataset != 'inference':
            raise ValueError(f"No ground truth provided. Dataset should be 'inference' only. Got {dataset}")
        self.dataset = dataset

        if name and not isinstance(name, str):
            raise TypeError(f'AOI name should be a string. Got {name} of type {type(name)}')
        elif not name:
            # Defaults to name of image without suffix
            name = self.img.stem
        self.name = name

        if index and not isinstance(index, int):
            raise TypeError(f'AOI index should be a integer. Got {index} of type {type(index)}')
        self.index = index

        if tiles_dir and not isinstance(tiles_dir, (Path, str)):
            raise TypeError(f'Experiment directory should be of class pathlib.Path or string.\n'
                            f'Got {tiles_dir} of type {type(tiles_dir)}')
        self.tiles_dir = tiles_dir / self.dataset.strip() / self.name.strip()
        self.tiles_dir_img = self.tiles_dir / 'images'
        self.tiles_dir_gt = self.tiles_dir / 'labels'
        self.tiles_pairs_list = []

    @classmethod
    def from_dict(cls, aoi_dict, tiles_dir: Union[Path, str] = None, attr_field: str = None,
                  attr_vals: list = None, index: int = None):
        if not isinstance(aoi_dict, dict):
            raise TypeError('Input data should be a list of dictionaries.')
        if not {'tif', 'meta', 'gpkg', 'attribute_name', 'dataset'}.issubset(set(aoi_dict.keys())):
            raise ValueError(f"Input data should minimally contain the following keys: \n"
                             f"'tif', 'meta', 'gpkg', 'attribute_name', 'dataset'.")
        if not aoi_dict['gpkg']:
            logging.warning(f"No ground truth data found for {aoi_dict['tif']}.\n"
                            f"Only imagery will be processed from now on")
        if not "aoi" not in aoi_dict.keys() or aoi_dict['aoi']:
            aoi_dict['aoi'] = Path(aoi_dict['tif']).stem
        # attribute field will be set from tiler if attribute field in csv is empty
        aoi_dict['attribute_name'] = attr_field if not aoi_dict['attribute_name'] else aoi_dict['attribute_name']
        new_aoi = cls(img=aoi_dict['tif'],
                      gt=aoi_dict['gpkg'],
                      dataset=aoi_dict['dataset'],
                      attr_field=aoi_dict['attribute_name'],
                      attr_vals=attr_vals,
                      name=aoi_dict['aoi'],
                      index=index,
                      tiles_dir=tiles_dir)
        return new_aoi

    @staticmethod
    def filter_gdf_by_attribute(gdf_tile: Union[str, Path, gpd.GeoDataFrame], attr_field: str = None,
                                attr_vals: Sequence = None):
        """
        Filter features from a geopandas.GeoDataFrame according to an attribute field and filtering values
        @param gdf_tile: str, Path or gpd.GeoDataFrame
            GeoDataFrame or path to GeoDataFrame to filter feature from
        @return: Subset of source GeoDataFrame with only filtered features (deep copy)
        """
        gdf_tile = _check_gdf_load(gdf_tile)
        if not attr_field:
            return gdf_tile, None
        elif attr_field and not attr_vals:
            raise ValueError(f'No attribute values given to filter features for attribute field "{attr_field}". '
                             f'If all values are to be kept, these values should be listed in attribute values '
                             f'in the dataset configuration')
        if not attr_field in gdf_tile.columns:
            attr_field = attr_field.split('/')[-1]
        if gdf_tile.empty:
            return gdf_tile, attr_field
        try:
            condList = [gdf_tile[f'{attr_field}'] == val for val in attr_vals]
            condList.extend([gdf_tile[f'{attr_field}'] == str(val) for val in attr_vals])
            allcond = functools.reduce(lambda x, y: x | y, condList)  # combine all conditions with OR
            gdf_filtered = gdf_tile[allcond].copy(deep=True)
            logging.debug(f'\nSuccessfully filtered features from GeoDataFrame"\n'
                          f'Filtered features: {len(gdf_filtered)}\n'
                          f'Total features: {len(gdf_tile)}\n'
                          f'Attribute field: "{attr_field}"\n'
                          f'Filtered values: {attr_vals}')
            if gdf_filtered.empty:
                logging.warning(f'\nFeatures are present for given attribute field "{attr_field}", but none with values'
                                f'{attr_vals}. Values present: {gdf_tile[attr_field].unique()}')
            return gdf_filtered, attr_field
        except KeyError as e:
            logging.critical(f'\nNo attribute named {attr_field} in GeoDataFrame. \n'
                             f'If all geometries should be kept, leave "attr_field" and "attr_vals" blank.\n'
                             f'Attributes: {gdf_tile.columns}\n'
                             f'GeoDataFrame: {gdf_tile.info()}')
            raise e

    @staticmethod
    def annot_percent(img_tile: Union[str, Path, rasterio.DatasetReader],
                      gdf_tile: Union[str, Path, gpd.GeoDataFrame],
                      tile_bounds: Polygon = None):
        """
        Calculate percentage of values in GeoDataFrame that contain classes other than background
        @param img_tile: str, Path or rasterio.DatasetReader
        @param gdf_tile: str, Path or gpd.GeoDataFrame
        @return: (int) Annotated percent
        """
        gdf_tile = _check_gdf_load(gdf_tile)
        if gdf_tile.empty:
            return 0
        img_tile_dataset = _check_rasterio_im_load(img_tile)
        if not tile_bounds:
            crs_match, img_crs, gt_crs = assert_crs_match(img_tile, gdf_tile)
            if not crs_match:
                tile_bounds = reproject_geometry(box(*img_tile_dataset.bounds),
                                                 input_crs=img_crs,
                                                 target_crs=gt_crs)
            else:
                tile_bounds = box(*img_tile_dataset.bounds)

        annot_ct_vec = gdf_tile.area.sum()
        annot_perc = annot_ct_vec / tile_bounds.area
        return annot_perc * 100


class Tiler(object):
    def __init__(self,
                 experiment_root_dir: Union[Path, str],
                 src_data_by_index: dict = None,
                 tile_size: int = 1024,
                 tile_stride: int = None,
                 resizing_factor: Union[int, float] = 1,
                 min_annot_perc: int = 0,
                 num_bands: int = None,
                 bands_idxs: Sequence = None,
                 min_img_tile_size: int = None,
                 val_percent: int = None,
                 attr_field_exp: str = None,
                 attr_vals_exp: list = None,
                 debug: bool = False):
        """
        @param experiment_root_dir: pathlib.Path or str
            Root directory under which all tiles will written (in subfolders)
        @param src_data_by_index: dictionary
            Dictionary where keys are indices and values are objects of class AOI.
            AOI objects contain properties including paths of source data and other data-related info.
        @param tile_size: int, optional
            Size of tiles to output. Defaults to 1024
        @param tile_stride: int, optional
            Number of pixels between each tile. Defaults to tile_size
        @param resizing_factor: int or float, optional
            Multiple by which source imagery must be resampled. Destination size must be divisible by this multiple
            without remainder. Rasterio will use bilinear resampling. Defaults to 1 (no resampling).
        @param min_annot_perc: int, optional
            If ground truth tile above this minimum annotated percentage,
            the gt tile will be kept in final dataset
        @param num_bands: int, optional
            Number of bands of imagery. If not provided, all bands will be included
        @param bands_idxs: list, optional
            The list of channel indices to be included in the output array.
            If not provided, all channels will be included. *Note:* per
            ``rasterio`` convention, indexing starts at ``1``, not ``0``.
        @param min_img_tile_size: integer, optional
            Minimum size in bytes to include imagery tile in training/testing dataset.
            If not provided, all tiles will be kept.
        @param val_percent: integer, optional
            Percentage of training tiles that should be written to validation set
        @param attr_field_exp: str, optional
            Attribute field for ground truth tiles across entire experiment from which features will filtered based on
            values provided
        @param attr_vals_exp: list, optional
            Values from attribute field for which features will be kept. These values must apply across entire dataset
        """
        if src_data_by_index and not isinstance(src_data_by_index, dict):
            raise TypeError('Input data should be a List')
        self.src_data_dict = src_data_by_index
        if not isinstance(experiment_root_dir, (Path, str)):
            raise TypeError(f'Tiles root directory should be a of class pathlib.Path or a string.\n'
                            f'Got {experiment_root_dir} of type {type(experiment_root_dir)}')
        if not Path(experiment_root_dir).is_dir():
            raise FileNotFoundError(f'{experiment_root_dir} is not a valid directory')
        self.tiles_root_dir = Path(experiment_root_dir)

        if not isinstance(tile_size, int):
            raise TypeError(f'Tile size should be an integer. Got {tile_size} of type {type(tile_size)}')
        self.dest_tile_size = tile_size

        # Tile stride defaults to tile size
        if not tile_stride:
            tile_stride = self.dest_tile_size
        if not isinstance(tile_stride, int):
            raise TypeError(f'Tile stride should be an integer. Got {tile_stride} of type {type(tile_stride)}')
        self.tile_stride = tile_stride

        if not isinstance(resizing_factor, (int, float)):
            raise TypeError(f'Resizing factor should be an integer or float.\n'
                            f'Got {resizing_factor} of type {type(resizing_factor)}')
        if resizing_factor is not None and self.dest_tile_size % resizing_factor != 0:
            raise ValueError(f'Destination tile size "{self.dest_tile_size}" must be divisible by resize '
                             f'"{resizing_factor}"')
        self.resizing_factor = resizing_factor

        if not isinstance(min_annot_perc, int) and 0 <= min_annot_perc <= 100:
            raise TypeError(f'Minimum annotated percent should be an integer between 0 and 100.\n'
                            f'Got {min_annot_perc} of type {type(min_annot_perc)}')
        self.min_annot_perc = min_annot_perc

        if num_bands and not isinstance(num_bands, int):
            raise TypeError(f'Number of bands should be an integer.\n'
                            f'Got {num_bands} of type {type(num_bands)}')
        elif not num_bands:
            logging.warning(f'Number of bands not defined. Defaulting to number of bands in imagery.')
            num_bands_set = set([aoi.meta['count'] for aoi in src_data_by_index.values()])
            if len(num_bands_set) > 1:
                raise ValueError(f'Not all imagery has equal number of bands. '
                                 f'Check imagery or define bands indexes to keep. \n'
                                 f'Number of bands found: {num_bands_set}')
        self.num_bands = num_bands

        self.tiles_dir_name = self.make_tiles_dir_name(self.dest_tile_size, self.num_bands)

        self.tiles_root_dir = experiment_root_dir / self.tiles_dir_name
        if self.tiles_root_dir.is_dir():
            logging.warning(f'Tiles root directory exists: {self.tiles_root_dir}.\n'
                            f'Make sure samples belong to the same experiment.')
        Path.mkdir(self.tiles_root_dir, exist_ok=True, parents=True)
        logging.info(f'Tiles will be written to {self.tiles_root_dir}\n\n')

        if bands_idxs and not isinstance(bands_idxs, Sequence) and self.num_bands == len(bands_idxs):
            raise TypeError(f"Bands indexes should be a list of same length as number of bands ({self.num_bands}).\n"
                            f"Bands_idxs: {bands_idxs}\n"
                            f"num_bands: {num_bands}")
        elif bands_idxs and 0 in bands_idxs:
            raise ValueError(f'As per rasterio convention, band indexing starts at 1, not 0')
        self.bands_idxs = bands_idxs

        # FIXME: refactor. The name suggest it's the size as shape not as bytes. Or remove all together...
        if min_img_tile_size and not isinstance(min_img_tile_size, int):
            raise TypeError(f'Minimum image size should be an integer.\n'
                            f'Got {min_img_tile_size} of type {type(min_img_tile_size)}')
        self.min_img_tile_size = min_img_tile_size

        if val_percent and not isinstance(val_percent, int):
            raise TypeError(f'Validation percentage should be an integer.\n'
                            f'Got {val_percent} of type {type(val_percent)}')
        self.val_percent = val_percent

        self.with_gt = True

        if attr_field_exp and not isinstance(attr_field_exp, str):
            raise TypeError(f'Attribute field name should be a string.\n'
                            f'Got {attr_field_exp} of type {type(attr_field_exp)}')
        self.attr_field_exp = attr_field_exp

        if attr_vals_exp and isinstance(attr_vals_exp, int):
            attr_vals_exp = [attr_vals_exp]
        elif attr_vals_exp and not isinstance(attr_vals_exp, Sequence):
            raise TypeError(f'Attribute values should be a tuple or list.\n'
                            f'Got {attr_vals_exp} of type {type(attr_vals_exp)}')
        self.attr_vals_exp = attr_vals_exp

        self.debug = debug

    def with_gt_checker(self):
        for aoi in self.src_data_dict.values():
            if not aoi.gt:
                self.with_gt = False
                logging.warning(f"No ground truth data found for {aoi.img}. Only imagery will be processed from now on")

    # TODO: add from glob pattern
    # TODO: add from band separated imagery
    def aois_from_csv(self, csv_path):
        """
        Instantiate a Tiler object from a csv containing list of input data.
        See README for details on expected structure of csv.
        @param csv_path: path to csv file
        @return: Tiler instance
        """
        aois = {}
        data_list = read_csv(csv_path)
        logging.info(f'\n\tSuccessfully read csv file: {Path(csv_path).name}\n'
                     f'\tNumber of rows: {len(data_list)}\n'
                     f'\tCopying first row:\n{data_list[0]}\n')
        for i, aoi_dict in tqdm(enumerate(data_list), desc="Creating AOI's"):
            try:
                new_aoi = AOI.from_dict(aoi_dict=aoi_dict,
                                        tiles_dir=self.tiles_root_dir,
                                        attr_field=self.attr_field_exp,
                                        attr_vals=self.attr_vals_exp,
                                        index=i)
                aois[new_aoi.index] = new_aoi
            except FileNotFoundError as e:
                logging.critical(f"{e}\nGround truth file may not exist or is empty.\n"
                                 f"Failed to create AOI:\n{aoi_dict}\n"
                                 f"Index: {i}")
        return aois

    @staticmethod
    def make_tiles_dir_name(tile_size, num_bands):
        return f'tiles{tile_size}_{num_bands}bands'

    @staticmethod
    def make_dataset_file_name(exp_name: str, min_annot: int, dataset: str, attr_vals: Sequence = None):
        if isinstance(attr_vals, int):
            attr_vals = [attr_vals]
        vals = "_feat" + "-".join([str(val) for val in attr_vals]) if attr_vals else ""
        min_annot_str = f"_min-annot{min_annot}"
        sampling_str = vals + min_annot_str
        # TODO: should this be a csv or txt?
        dataset_file_name = f'{exp_name}{sampling_str}_{dataset}.txt'
        return dataset_file_name, sampling_str

    @staticmethod
    def find_gt_tile_match(img_tile_path: Union[Path, str], dest_gt_tiles_dir: Union[Path, str]):
        """
        Find a ground truth tile matching a given imagery tile
        @param img_tile_path: path to imagery tile
        @param dest_gt_tiles_dir: path to destination ground truth tiles
        @return: path to matching ground truth tile, if no more and no less than one gt tile is found.
        """
        gt_tile_splits = img_tile_path.stem.split('_')
        img_tile_prefix = "_".join(gt_tile_splits[:-2])
        gt_glob_pat = f'{img_tile_prefix}_{gt_tile_splits[-2]}*_{gt_tile_splits[-1]}*.geojson'
        logging.debug(f'Finding ground truth tile to match imagery:\n'
                      f'Image tile {img_tile_path}\n'
                      f'Destination ground truth directory: {dest_gt_tiles_dir}\n'
                      f'Glob pattern to find ground truth tile: {gt_glob_pat}\n')
        gt_tile_glob = list(dest_gt_tiles_dir.glob(gt_glob_pat))
        if len(gt_tile_glob) == 1:
            gt_tile = gt_tile_glob[0]
        elif len(gt_tile_glob) > 1:
            raise ValueError(f'There should be only one ground truth tile for each imagery tile.\n'
                             f'Got {len(gt_tile_glob)}.\n'
                             f'Imagery tile: {img_tile_path}\n'
                             f'Ground truth match: {gt_tile_glob}')
        else:
            return None

        if gt_tile.is_file():
            return gt_tile
        else:
            raise FileNotFoundError(f'Missing ground truth tile {gt_tile}')

    def tiling_checker(self,
                       src_img: Union[str, Path],
                       dest_img_tiles_dir: Union[str, Path],
                       dest_gt_tiles_dir: Union[str, Path] = None,
                       verbose: bool = True):
        """
        Checks how many tiles should be created and compares with number of tiles already written to output directory
        @param src_img: path to source image
        @param dest_img_tiles_dir: optional, path to output directory where imagery tiles will be created
        @param dest_gt_tiles_dir: optional, path to output directory where ground truth tiles will be created
        @param tile_size: (int) optional, size of tile. Defaults to 1024.
        @param tile_stride: (int) optional, stride to use during tiling. Defaults to tile_size.
        @return: number of actual tiles in output directory, number of expected tiles
        """
        act_data_tiles = []
        metadata = _check_rasterio_im_load(src_img).meta
        dest_width = metadata['width'] * self.resizing_factor
        dest_height = metadata['height'] * self.resizing_factor
        tiles_x = 1 + math.ceil((dest_width - self.dest_tile_size) / self.tile_stride)
        tiles_y = 1 + math.ceil((dest_height - self.dest_tile_size) / self.tile_stride)
        nb_exp_tiles = tiles_x * tiles_y
        # glob for tiles of the vector ground truth if 'geojson' is in the suffix
        act_img_tiles = list(dest_img_tiles_dir.glob(f'{Path(src_img).stem}*.tif'))
        nb_act_img_tiles = len(act_img_tiles)
        nb_act_gt_tiles = 0
        if dest_gt_tiles_dir:
            # check if all imagery tiles have a matching ground truth tile
            for img_tile in act_img_tiles:
                gt_tile = self.find_gt_tile_match(img_tile, dest_gt_tiles_dir)
                # if no errors are raised, then the gt tile was found, we can increment our counter
                if gt_tile:
                    act_data_tiles.append((img_tile, gt_tile))
                    nb_act_gt_tiles += 1
        else:
            act_data_tiles = [(img_tile, None) for img_tile in act_img_tiles]

        if verbose:
            logging.info(f'Number of actual imagery tiles : {nb_act_img_tiles}\n'
                         f'Number of actual ground truth tiles : {nb_act_gt_tiles}\n'
                         f'Number of expected tiles : {nb_exp_tiles}\n')
        return act_data_tiles, nb_act_img_tiles, nb_act_gt_tiles, nb_exp_tiles

    def get_src_tile_size(self):
        """
        Sets outputs dimension of source tile if resizing, given destination size and resizing factor
        @param dest_tile_size: (int) Size of tile that is expected as output
        @param resize_factor: (float) Resize factor to apply to source imagery before outputting tiles
        @return: (int) Source tile size
        """
        if self.resizing_factor:
            return int(self.dest_tile_size / self.resizing_factor)
        else:
            return self.dest_tile_size

    def tiling_per_aoi(self, aoi: AOI):
        """
        Calls solaris_gdl tiling function and outputs tiles in output directories
        @param src_img: path to source image
        @param out_img_dir: path to output tiled images directory
        @param out_label_dir: optional, path to output tiled images directory
        @param src_label: optional, path to source label (must be a geopandas compatible format like gpkg or geojson)
        @return: written tiles to output directories as .tif for imagery and .geojson for label.
        """
        self.src_tile_size = self.get_src_tile_size()
        raster_tiler = tile.raster_tile.RasterTiler(dest_dir=aoi.tiles_dir_img,
                                                    src_tile_size=(self.src_tile_size, self.src_tile_size),
                                                    dest_tile_size=(self.dest_tile_size, self.dest_tile_size),
                                                    resize=self.resizing_factor,
                                                    alpha=False,
                                                    verbose=True)
        raster_bounds_crs = raster_tiler.tile(aoi.img, channel_idxs=self.bands_idxs)
        logging.debug(f'Raster bounds crs: {raster_bounds_crs}\n'
                      f'')

        if self.with_gt:
            dest_crs = raster_tiler.dest_crs if raster_tiler.dest_crs.is_epsg_code else None
            vec_tler = tile.vector_tile.VectorTiler(dest_dir=aoi.tiles_dir_gt,
                                                    dest_crs=dest_crs,
                                                    verbose=True,
                                                    super_verbose=self.debug)
            vec_tler.tile(src=aoi.gt,
                          tile_bounds=raster_tiler.tile_bounds,
                          tile_bounds_crs=raster_bounds_crs,
                          dest_fname_base=aoi.img.stem)

        else:
            return aoi, raster_tiler.tile_paths, None, None
        return aoi, raster_tiler.tile_paths, vec_tler.tile_paths, vec_tler.tile_bds_reprojtd

    def filter_tile_pair(self, aoi: AOI, img_tile: Union[str, Path],
                         gt_tile: Union[str, Path, gpd.GeoDataFrame], tile_bounds: Polygon = None):
        map_img_gdf = _check_gdf_load(gt_tile)
        # Check if size of image tile reaches size threshold, else continue
        sat_size = img_tile.stat().st_size
        annot_perc = aoi.annot_percent(img_tile, map_img_gdf, tile_bounds)
        # First filter by size
        if sat_size < self.min_img_tile_size:
            logging.debug(f'File {aoi.img} below minimum size ({self.min_img_tile_size}): {sat_size}')
            return False, sat_size, annot_perc
        # Then filter by annotated percentage
        if aoi.dataset == 'tst' or annot_perc >= self.min_annot_perc:
            return True, sat_size, annot_perc
        elif aoi.dataset == 'trn' and annot_perc < self.min_annot_perc:
            logging.debug(f"Ground truth tile in training dataset doesn't reach minimum annotated percentage.\n"
                          f"Ground truth tile: {gt_tile}\n"
                          f"Annotated percentage: {annot_perc}\n"
                          f"Minimum annotated percentage: {self.min_annot_perc}")
        return False, sat_size, annot_perc

    def get_burn_gt_tile_path(self, attr_vals: Sequence, gt_tile: Union[str, Path]):
        _, samples_str = self.make_dataset_file_name(None, self.min_annot_perc, None, attr_vals)
        out_burned_gt_path = Path(gt_tile).parent.parent / 'labels_burned' / f'{Path(gt_tile).stem}{samples_str}.tif'
        out_burned_gt_path.parent.mkdir(exist_ok=True)
        return out_burned_gt_path

    def burn_gt_tile(self, aoi: AOI,
                     img_tile: Union[str, Path],
                     gt_tile: Union[gpd.GeoDataFrame, str, Path],
                     out_px_mask: Union[str, Path],
                     continuous: bool = True,
                     save_preview: bool = True,
                     dry_run: bool = False):
        """
        Return line to be written to a dataset file
        @param aoi: AOI object
        @param img_tile: str or pathlib.Path
            Path to image tile
        @param gt_tile: str, pathlib.Path or gpd.GeoDataFrame
            Path to ground truth tile or gpd.GeoDataFrame of ground truth
        @param out_px_mask: Burned tile output path
        @param continuous: bool, if True, burn values will be continuous starting at 1 for easier use in training an ML
                           model
        @param save_preview: bool, if True, a copy of the burned label will be created for quick preview from a file
                             manager. Burn values will be stretched to 255.
        @return:
        """

        if out_px_mask.is_file():
            logging.info(f'Burned ground truth tile exists: {out_px_mask}')
            return
        if not aoi.attr_field and aoi.attr_vals is not None:
            raise ValueError(f'Values for an attribute field have been provided, but no attribute field is set.\n'
                             f'Attribute values: {aoi.attr_vals}')
        elif aoi.attr_field is not None and not aoi.attr_vals:
            raise ValueError(f'An attribute field has been provided, but no attribute values were set.\n'
                             f'Attribute field: {aoi.attr_field}. If all values from attribute fields are '
                             f'to be kept, please input full list of values in dataset configuration.')
        # Burn value of attribute field from which features are being filtered. If single value is filtered
        # burn 255 value (easier for quick visualization in file manager)
        if not dry_run:
            burn_field = aoi.attr_field if aoi.attr_field else None
            burn_val = None if aoi.attr_field and aoi.attr_vals else 255
            if gt_tile.empty:
                burn_field = None
            elif continuous and aoi.attr_field:
                # Define new column 'burn_val' with continuous values for use during burning
                cont_vals_dict = {src: dst+1 for dst, src in enumerate(aoi.attr_vals)}
                if all(isinstance(val, str) for val in gt_tile[aoi.attr_field].unique().tolist()):
                    cont_vals_dict = {str(src): dst for src, dst in cont_vals_dict.items()}
                gt_tile['burn_val'] = gt_tile[aoi.attr_field].map(cont_vals_dict)
                burn_field = 'burn_val'  # overwrite burn_field
            elif continuous:
                burn_val = 1
            # burn to raster
            vector.mask.footprint_mask(df=gt_tile, out_file=str(out_px_mask),
                                       reference_im=str(img_tile),
                                       burn_field=burn_field,
                                       burn_value=burn_val)
            if save_preview:
                # burn preview to raster iin dedicated folder
                prev_out_px_mask = Path(f'{out_px_mask.parent}_preview') / f'{out_px_mask.stem}.png'
                prev_out_px_mask.parent.mkdir(exist_ok=True)
                with rasterio.open(out_px_mask) as burned_tile:
                    burned_tile_array = burned_tile.read()[0, ...]
                    matplotlib.pyplot.imsave(prev_out_px_mask, burned_tile_array)

    def filter_and_burn_dataset(self, aoi: AOI, img_tile: Union[str, Path],
                                gt_tile: Union[str, Path],
                                tile_bounds: Polygon = None,
                                continuous_vals: bool = True,
                                save_preview_labels: bool = False,
                                dry_run: bool = False):
        out_gt_burned_path = self.get_burn_gt_tile_path(attr_vals=aoi.attr_vals, gt_tile=gt_tile)
        # returns corrected attr_field if original field needed truncating
        gdf_tile, aoi.attr_field = aoi.filter_gdf_by_attribute(gdf_tile=gt_tile,
                                                               attr_field=aoi.attr_field,
                                                               attr_vals=aoi.attr_vals)
        keep_tile_pair, sat_size, annot_perc = self.filter_tile_pair(aoi,
                                                                     img_tile=img_tile,
                                                                     gt_tile=gdf_tile,
                                                                     tile_bounds=tile_bounds)
        logging.debug(annot_perc)
        random_val = np.random.randint(1, 100)
        # for trn tiles, sort between trn and val based on random number
        dataset = 'val' if aoi.dataset == 'trn' and random_val < self.val_percent else aoi.dataset
        if keep_tile_pair:
            self.burn_gt_tile(aoi,
                              img_tile=img_tile,
                              gt_tile=gdf_tile,
                              out_px_mask=out_gt_burned_path,
                              continuous=continuous_vals,
                              save_preview=save_preview_labels,
                              dry_run=dry_run)
            dataset_line = f'{img_tile.absolute()};{out_gt_burned_path.absolute()};{round(annot_perc)}\n'
            return (dataset, dataset_line)
        else:
            return (dataset, None)


def gt_from_img(img_path, gt_dir_rel2img):
    image = Path(img_path)
    gt_dir = image.parent / gt_dir_rel2img
    if not gt_dir.is_dir():
        logging.warning(f'Failed to find ground truth directory for image:\n'
                        f'{image}\n'
                        f'Ground truth directory should be: {gt_dir}')
        return
    gts = list(gt_dir.iterdir())
    gts_names = [str(gt.name) for gt in gts]
    gt_matches = difflib.get_close_matches(image.stem, gts_names)
    if not gt_matches:
        raise FileNotFoundError(f"Couldn't find a ground truth file to match imagery:\n"
                                f"{image}")
    gt = gt_dir / gt_matches[0]
    csv_line = (image, gt.resolve(), 'trn')
    return csv_line


def csv_from_glob(img_glob, gt_dir_rel2img, parallel=False):
    """
    Write a GDL csv from glob patterns to imagery and ground truth data
    @param img_glob: glob pattern to imagery
    @param gt_dir_rel2img: ground truth directory relative to imagery
    @return: path to output csv
    """
    csv_lines = []
    # 10 next lines from: https://github.com/WongKinYiu/yolor/blob/main/utils/datasets.py
    p = str(Path(img_glob))  # os-agnostic
    p = os.path.abspath(p)  # absolute path
    if '*' in p:
        images = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        images = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        images = [p]  # files
    else:
        raise Exception('ERROR: %s does not exist' % p)
    logging.warning(f"Dataset will only be 'trn' when creating csv from glob")
    input_args = []
    for image in tqdm(images, desc=f'Searching for ground truth match to {len(images)} globbed images'):
        if parallel:
            input_args.append([gt_from_img, image, gt_dir_rel2img])
        else:
            csv_line = gt_from_img(image, gt_dir_rel2img)
            csv_lines.append(csv_line)

    if parallel:
        logging.info(f'Parallelizing search for ground truth for {len(images)} globbed images...')
        proc = multiprocessing.cpu_count()
        with multiprocessing.get_context('spawn').Pool(processes=proc) as pool:
            lines = pool.map_async(map_wrapper, input_args).get()
        csv_lines = lines

    # Export to pickle rather than csv?
    # TODO: https://towardsdatascience.com/stop-using-csvs-for-storage-here-are-the-top-5-alternatives-e3a7c9018de0
    out_csv = Path(img_glob.split('/*')[0]) / f'{Path(img_glob.split("*")[0]).stem}.csv'
    with open(out_csv, 'w') as out:
        write = csv.writer(out)
        write.writerows(csv_lines)
    logging.info(f'Finished glob. Wrote to csv: {out_csv}')
    return str(out_csv)


def map_wrapper(x):
    '''For multi-threading'''
    return x[0](*(x[1:]))


def main(cfg: DictConfig) -> None:
    """
    Function that create training, validation and testing datasets preparation.

    Process
    -------
    1. Read csv file and validate existence of all input files and GeoPackages.

    2. Do the following verifications:
        1. Assert number of bands found in raster is equal to desired number
           of bands.
        2. Check that `num_classes` is equal to number of classes detected in
           the specified attribute for each GeoPackage.
           Warning: this validation will not succeed if a Geopackage
                    contains only a subset of `num_classes` (e.g. 3 of 4).
        3. Assert Coordinate reference system between raster and gpkg match.

    3. For each line in the csv file, output tiles from imagery and label files based on "samples_size" parameter
    N.B. This step can be parallelized with multiprocessing. Tiling will be skipped if tiles already exist.

    4. Create pixels masks from each geojson tile and write a list of image tile / pixelized label tile to text file
    N.B. for train/val datasets, only tiles that pass the "min_annot_percent" threshold are kept.

    -------
    :param params: (dict) Parameters found in the yaml config file.
    """
    # PARAMETERS
    # TODO add the Victor module to manage the modalities
    in_bands = get_key_def('in_bands', cfg['dataset'], expected_type=Sequence)
    out_modalities = get_key_def('out_modalities', cfg['dataset'], expected_type=Sequence)
    num_bands = len(cfg.dataset.out_modalities)
    bands_idxs = select_modalities(in_bands=in_bands, out_modalities=out_modalities)
    debug = cfg.debug
    dry_run = cfg.dry_run

    # RAW DATA PARAMETERS
    data_path = get_key_def('raw_data_dir', cfg['dataset'], to_path=True, validate_path_exists=True)
    csv_file = get_key_def('raw_data_csv', cfg['dataset'], to_path=True, validate_path_exists=True)
    tiles_dir = get_key_def('tiles_data_dir', cfg['dataset'], default=data_path, to_path=True, validate_path_exists=True)

    # SAMPLE PARAMETERS
    samples_size = get_key_def('tile_size', cfg['tiling'], default=512, expected_type=int)
    min_annot_perc = get_key_def('min_annot_perc', cfg['tiling'], default=0)
    continuous_vals = get_key_def('continuous_values', cfg['tiling'], default=True)
    save_prev_labels = get_key_def('save_preview_labels', cfg['tiling'], default=False)
    resize = get_key_def('resampling', cfg['tiling'], default=1, expected_type=int)
    parallel = get_key_def('multiprocessing', cfg['tiling'], default=False, expected_type=bool)
    min_raster_tile_size = get_key_def('min_raster_tile_size', cfg['tiling'], default=0, expected_type=int)

    val_percent = int(get_key_def('train_val_percent', cfg['dataset'], default=0.3)['val'] * 100)
    validate_data = get_key_def('validate_data', cfg['dataset'], default=True, expected_type=bool)
    attr_field = get_key_def('attribute_field', cfg['dataset'], None, expected_type=str)
    attr_vals = get_key_def('attribute_vals', cfg['dataset'], None, expected_type=(Sequence, int))

    # ADD GIT HASH FROM CURRENT COMMIT TO PARAMETERS (if available and parameters will be saved to hdf5s).
    with open_dict(cfg):
        cfg.general.git_hash = get_git_hash()

    experiment_name = cfg.general.project_name
    exp_dir = data_path / experiment_name if not tiles_dir else tiles_dir
    if exp_dir.is_dir():
        print(f'WARNING: Data path exists: {exp_dir}. Make sure samples belong to the same experiment.')
    Path.mkdir(exp_dir, exist_ok=True, parents=True)

    if debug:
        logging.warning(f'Debug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    tiler = Tiler(experiment_root_dir=exp_dir,
                  tile_size=samples_size,
                  resizing_factor=resize,
                  min_annot_perc=min_annot_perc,
                  num_bands=num_bands,
                  bands_idxs=bands_idxs,
                  min_img_tile_size=min_raster_tile_size,
                  val_percent=val_percent,
                  attr_field_exp=attr_field,
                  attr_vals_exp=attr_vals,
                  debug=debug)
    tiler.src_data_dict = tiler.aois_from_csv(csv_path=csv_file)
    tiler.with_gt_checker()

    # VALIDATION: Assert number of bands in imagery is {num_bands}
    if validate_data:
        for aoi in tqdm(tiler.src_data_dict.values(),
                        desc=f'Asserting number of bands in imagery is {tiler.num_bands}'):
            validate_num_bands(raster=aoi.img, num_bands=tiler.num_bands, bands_idxs=tiler.bands_idxs)
    else:
        logging.warning(f"Skipping validation of imagery and correspondance between actual and expected number of "
                        f"bands. Use only is data has already been validated once.")

    datasets = ['trn', 'val', 'tst']

    # For each row in csv: (1) burn vector file to raster, (2) read input raster image, (3) prepare samples
    input_args = []
    tilers = []
    logging.info(f"Preparing samples \n\tSamples_size: {samples_size} ")
    for aoi in tqdm(tiler.src_data_dict.values(), position=0, leave=False):
        try:
            out_img_dir = aoi.tiles_dir / 'images'
            out_gt_dir = aoi.tiles_dir / 'labels' if tiler.with_gt else None
            do_tile = True
            data_pairs, nb_act_img_t, nb_act_gt_t, nb_exp_t = tiler.tiling_checker(aoi.img, out_img_dir, out_gt_dir)
            # add info about found tiles for each aoi by aoi's index
            tiler.src_data_dict[aoi.index].tiles_pairs_list = [(*data_pair, None) for data_pair in data_pairs]
            if nb_act_img_t == nb_exp_t == nb_act_gt_t or not tiler.with_gt and nb_act_img_t == nb_exp_t:
                logging.info('All tiles exist. Skipping tiling.\n')
                do_tile = False
            elif nb_act_img_t > nb_exp_t and nb_act_gt_t > nb_exp_t or \
                    not tiler.with_gt and nb_act_img_t > nb_exp_t:
                logging.error(f'\nToo many tiles for "{aoi.img}". \n'
                              f'Expected: {nb_exp_t}\n'
                              f'Actual image tiles: {nb_act_img_t}\n'
                              f'Actual label tiles: {nb_act_gt_t}\n'
                              f'Skipping tiling.')
                do_tile = False
            elif nb_act_img_t > 0 or nb_act_gt_t > 0:
                logging.error(f'Missing tiles for {aoi.img}. \n'
                              f'Expected: {nb_exp_t}\n'
                              f'Actual image tiles: {nb_act_img_t}\n'
                              f'Actual label tiles: {nb_act_gt_t}\n'
                              f'Starting tiling from scratch...')
            else:
                logging.debug(f'Expected: {nb_exp_t}\n'
                              f'Actual image tiles: {nb_act_img_t}\n'
                              f'Actual label tiles: {nb_act_gt_t}\n'
                              f'Starting tiling from scratch...')

            # if no previous step has shown existence of all tiles, then go on and tile.
            if do_tile and not dry_run:
                if parallel:
                    input_args.append([tiler.tiling_per_aoi, aoi])
                else:
                    try:
                        tiler_pair = tiler.tiling_per_aoi(aoi)
                        tilers.append(tiler_pair)
                    except ValueError as e:
                        logging.debug(f'Failed to tile\n'
                                      f'Img: {aoi.img}\n'
                                      f'GT: {aoi.gt}')
                        raise e
            elif dry_run:
                logging.warning(f'DRY RUN. No tiles will be written')

        except OSError:
            logging.exception(f'An error occurred while preparing samples with "{aoi.img.stem}" (tiff) and '
                              f'{aoi.gt.stem} (gpkg).')
            continue

    if parallel:
        logging.info(f'Will tile {len(input_args)} images and labels...')
        with multiprocessing.get_context('spawn').Pool(None) as pool:
            tilers = pool.map_async(map_wrapper, input_args).get()

    for aoi, raster_tiler_paths, vec_tler_paths, vec_tler_tbs in tqdm(tilers,
                                                                      desc=f"Updating AOIs' information about their tiles paths"):
        for img_tile_path, gt_tile_path, gt_tile_bds in zip(raster_tiler_paths, vec_tler_paths, vec_tler_tbs):
            tiler.src_data_dict[aoi.index].tiles_pairs_list.append((img_tile_path, gt_tile_path, gt_tile_bds))

    logging.info(f"Tiling done. Creating pixel masks from clipped geojsons...\n"
                 f"Validation set: {val_percent} % of created training tiles")
    # TODO: how does train_segmentation know where these are?
    dataset_files = {}
    for dset in datasets:
        name, _ = tiler.make_dataset_file_name(experiment_name, tiler.min_annot_perc, dset, tiler.attr_vals_exp)
        dset_path = tiler.tiles_root_dir / name
        if dset_path.is_file():
            logging.critical(f'Dataset list exists and will be overwritten: {dset_path}')
            dset_path.unlink()
        dataset_files[dset] = dset_path

    input_args = []
    dataset_lines = []
    datasets_total = {dataset: 0 for dataset in datasets}
    if not tiler.with_gt:
        logging.warning('List of training tiles contains no ground truth, only imagery.')
    # loop through line of csv again
    for aoi in tqdm(tiler.src_data_dict.values(), position=0,
                    desc='Looping in AOIs'):
        if debug:
            for img_tile, gt_tile, _ in tqdm(aoi.tiles_pairs_list,
                                             desc='DEBUG: Checking if data tiles are valid'):
                is_valid, _ = validate_raster(img_tile)
                if not is_valid:
                    logging.error(f'Invalid imagery tile: {img_tile}')
                try:
                    gpd.read_file(gt_tile)
                except Exception as e:
                    logging.error(f'Invalid ground truth tile: {img_tile}. Error: {e}')

        for img_tile, gt_tile, tbs in tqdm(aoi.tiles_pairs_list, position=1,
                                           desc=f'Filter {len(aoi.tiles_pairs_list)} tiles and burn ground truth'):
            datasets_total[aoi.dataset] += 1
            # If for inference, write only image tile since there's no ground truth
            if not tiler.with_gt:
                if gt_tile is not None:
                    logging.error(f'Tiler set without ground truth, but ground truth was found. It will be ignored.\n'
                                  f'Image tile: {img_tile}\n'
                                  f'Ground truth: {gt_tile}')
                dataset_line = f'{img_tile.absolute()}\n'
                dataset_lines.append((aoi.dataset, dataset_line))
            # if for train, validation or test dataset, then filter, burn and provided complete line to write to file
            else:
                if parallel:
                    input_args.append([tiler.filter_and_burn_dataset, aoi, img_tile, gt_tile, tbs, dry_run])
                else:
                    line_tuple = tiler.filter_and_burn_dataset(aoi,
                                                               img_tile=img_tile,
                                                               gt_tile=gt_tile,
                                                               tile_bounds=tbs,
                                                               continuous_vals=continuous_vals,
                                                               save_preview_labels=save_prev_labels,
                                                               dry_run=dry_run)
                    dataset_lines.append(line_tuple)

    if parallel:
        logging.info(f'Parallelizing burning of {len(input_args)} filtered ground truth tiles...')
        proc = multiprocessing.cpu_count()
        with multiprocessing.get_context('spawn').Pool(processes=proc) as pool:
            lines = pool.map_async(map_wrapper, input_args).get()
        dataset_lines.extend(lines)

    # write to dataset text file the data that was kept after filtering
    datasets_kept = {dataset: 0 for dataset in datasets}
    for line_tuple in tqdm(dataset_lines, desc=f"Writing {len(dataset_lines)} lines to dataset files"):
        dataset, dataset_line = line_tuple
        if dataset_line:
            with open(dataset_files[dataset], 'a') as dataset_file:
                dataset_file.write(dataset_line)
                datasets_kept[dataset] += 1

    # final report
    for dataset in datasets:
        if dataset == 'trn':
            logging.info(f"\nDataset: {dataset}\n"
                         f"Tiles kept (with non-zero values above {min_annot_perc}%): \n"
                         f"\t Train set: {datasets_kept[dataset]}\n"
                         f"\t Validation set: {datasets_kept['val']}\n"
                         f"Total tiles: {datasets_total[dataset]}\n"
                         f"Discarded tiles: {datasets_total[dataset] - datasets_kept['val'] - datasets_kept['trn']}")
        elif dataset == 'tst':
            logging.info(f"\nDataset: {dataset}\n"
                         f"Total tiles: {datasets_total[dataset]}\n")
    logging.info(f"End of process. See dataset files: \n")
    for dataset, file in dataset_files.items():
        logging.info(f"{dataset}: {str(file)}")