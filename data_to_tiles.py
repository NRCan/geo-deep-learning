import argparse
import functools
import math
import multiprocessing
from collections import OrderedDict
import logging
import logging.config
from datetime import datetime
from typing import List, Union

import numpy as np

from solaris_gdl.utils.core import _check_rasterio_im_load
from utils.logger import set_logging

np.random.seed(1234)  # Set random seed for reproducibility
import rasterio
import time

from pathlib import Path
from tqdm import tqdm
import geopandas as gpd

from utils.utils import get_key_def, read_csv, get_git_hash, map_wrapper
from utils.readers import read_parameters
from utils.verifications import validate_raster, validate_num_bands
from solaris_gdl import tile
from solaris_gdl import vector

logging.getLogger(__name__)


class AOI(object):
    """
    Object containing all data information about a single area of interest
    """
    def __init__(self, img: Union[Path, str],
                 dataset: str,
                 gt: Union[Path, str] = None,
                 img_meta: dict = None,
                 attr_field: str = None,
                 attr_vals: str = None,
                 name: str = None,
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
        if not Path(gt).is_file():
            raise FileNotFoundError(f'{gt} is not a valid file')
        self.gt = Path(gt)
        
        # TODO: original implementation may have expected a file directly
        if img_meta and not isinstance(img_meta, dict):
            raise TypeError(f'Image metadata should be a dictionary.\n'
                            f'Got {img_meta} of type {type(img_meta)}')
        elif not img_meta:
            # Defaults to image metadata as given by rasterio
            img_meta = _check_rasterio_im_load(self.img).meta
        self.img_meta = img_meta
        
        if gt and attr_field and not isinstance(attr_field, str):
            raise TypeError(f'Attribute field name should be a string.\n'
                            f'Got {attr_field} of type {type(attr_field)}')
        self.attr_field = attr_field

        if gt and attr_vals and not isinstance(attr_vals, list):
            raise TypeError(f'Attribute values should be a list.\n'
                            f'Got {attr_vals} of type {type(attr_vals)}')
        self.attr_vals = attr_vals
        
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
        
        if tiles_dir and not isinstance(tiles_dir, (Path, str)):
            raise TypeError(f'Experiment directory should be of class pathlib.Path or string.\n'
                            f'Got {tiles_dir} of type {type(tiles_dir)}')
        self.tiles_dir = tiles_dir / self.dataset.strip() / self.name.strip()
        self.tiles_dir_img = self.tiles_dir / 'sat_img'
        self.tiles_dir_gt = self.tiles_dir / 'map_img'
        self.tiles_pairs_list = []

    @classmethod
    def from_dict(cls, aoi_dict, tiles_dir: Union[Path, str] = None, attr_vals: list = None):
        if not isinstance(aoi_dict, dict):
            raise TypeError('Input data should be a list of dictionaries.')
        if not {'tif', 'meta', 'gpkg', 'attribute_name', 'dataset', 'aoi'}.issubset(set(aoi_dict.keys())):
            raise ValueError(f"Input data should minimally contain the following keys: \n"
                             f"'tif', 'meta', 'gpkg', 'attribute_name', 'dataset', 'aoi'.")
        if not aoi_dict['gpkg']:
            logging.warning(f"No ground truth data found for {aoi_dict['tif']}.\n"
                            f"Only imagery will be processed from now on")
        if not aoi_dict['aoi']:
            aoi_dict['aoi'] = Path(aoi_dict['tif']).stem
        new_aoi = cls(img=aoi_dict['tif'],
                      gt=aoi_dict['gpkg'],
                      dataset=aoi_dict['dataset'],
                      img_meta=aoi_dict['meta'],
                      attr_field=aoi_dict['attribute_name'],
                      attr_vals=attr_vals,
                      name=aoi_dict['aoi'],
                      tiles_dir=tiles_dir)
        return new_aoi


class Tiler(object):
    def __init__(self,
                 experiment_root_dir: Union[Path, str],
                 src_data_list: list = None,
                 tile_size: int = 1024,
                 tile_stride: int = None,
                 resizing_factor: Union[int, float] = 1,
                 min_annot_perc: int = 0,
                 num_bands: int = None,
                 bands_idxs: List = None):
        """
        @param experiment_root_dir: pathlib.Path or str
            Root directory under which all tiles will written (in subfolders)
        @param src_data_list: list of objects of class AOI
            List of AOI objects referring to paths of source data and other data-related info.
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
        """
        if src_data_list and not isinstance(src_data_list, list):
            raise TypeError('Input data should be a List')
        self.src_data_list = src_data_list
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
            num_bands_set = set([aoi.meta['count'] for aoi in src_data_list])
            if len(num_bands_set) > 1:
                raise ValueError(f'Not all imagery has equal number of bands. '
                                 f'Check imagery or define bands indexes to keep. \n'
                                 f'Number of bands found: {num_bands_set}')
        self.num_bands = num_bands

        self.tiles_dir_name = self.make_tiles_dir_name(self.dest_tile_size, self.min_annot_perc, self.num_bands)

        self.tiles_root_dir = experiment_root_dir / self.tiles_dir_name
        if self.tiles_root_dir.is_dir():
            logging.warning(f'Tiles root directory exists: {self.tiles_root_dir}.\n'
                            f'Make sure samples belong to the same experiment.')
        Path.mkdir(self.tiles_root_dir, exist_ok=True, parents=True)
        logging.info(f'Tiles will be written to {self.tiles_root_dir}\n\n')

        if bands_idxs and not isinstance(bands_idxs, list) and self.num_bands == len(bands_idxs):
            raise TypeError(f"Bands indexes should be a list of same length as number of bands ({self.num_bands}).\n"
                            f"Bands_idxs: {bands_idxs}\n"
                            f"num_bands: {num_bands}")
        elif bands_idxs and 0 in bands_idxs:
            raise ValueError(f'Per rasterio convention, band indexing starts at 1, not 0')
        self.bands_idxs = bands_idxs

        self.with_gt = True

    def with_gt_checker(self):
        for aoi in self.src_data_list:
            if not aoi.gt:
                self.with_gt = False
                logging.warning(f"No ground truth data found for {aoi.img}. Only imagery will be processed from now on")

    def aois_from_csv(self, csv_path, attr_vals):
        """
        Instantiate a Tiler object from a csv containing list of input data. 
        See README for details on expected structure of csv. 
        @param csv_path: path to csv file 
        @return: Tiler instance
        """
        aois = []
        data_list = read_csv(csv_path)
        logging.info(f'\n\tSuccessfully read csv file: {Path(csv_path).name}\n'
                     f'\tNumber of rows: {len(data_list)}\n'
                     f'\tCopying first row:\n{data_list[0]}\n')
        for aoi_dict in data_list:
            new_aoi = AOI.from_dict(aoi_dict=aoi_dict, tiles_dir=self.tiles_root_dir, attr_vals=attr_vals)
            aois.append(new_aoi)

        return aois

    @staticmethod
    def make_tiles_dir_name(tile_size, min_annot_perc, num_bands):
        return f'tiles{tile_size}_min-annot{min_annot_perc}_{num_bands}bands'
    
    @staticmethod
    def find_gt_tile_match(img_tile_path: Union[Path, str], dest_gt_tiles_dir: Union[Path, str]):
        """
        Find a ground truth tile matching a given imagery tile
        @param img_tile_path: path to imagery tile
        @param dest_gt_tiles_dir: path to destination ground truth tiles
        @return: path to matching ground truth tile, if no more and no less than one gt tile is found.
        """
        gt_tile_splits = img_tile_path.stem.split('_')
        gt_tile_glob = list(dest_gt_tiles_dir.glob(f'{gt_tile_splits[0]}_{gt_tile_splits[-2]}*_{gt_tile_splits[-1]}*.geojson'))
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
        metadata = _check_rasterio_im_load(src_img).meta
        tiles_x = 1 + math.ceil((metadata['width'] - self.dest_tile_size) / self.tile_stride)
        tiles_y = 1 + math.ceil((metadata['height'] - self.dest_tile_size) / self.tile_stride)
        nb_exp_tiles = tiles_x * tiles_y * self.resizing_factor**2
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
                    nb_act_gt_tiles += 1

        if verbose:
            logging.info(f'Number of actual imagery tiles : {nb_act_img_tiles}\n'
                         f'Number of actual ground truth tiles : {nb_act_gt_tiles}\n'
                         f'Number of expected tiles : {nb_exp_tiles}\n')
        return nb_act_img_tiles, nb_act_gt_tiles, nb_exp_tiles

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
        # FIXME: this list won't build if tiles are already created (tiling_per_aoi only called if they don't exist)
        aoi.tiles_pairs_list = [(img_tile_path, None) for img_tile_path in sorted(raster_tiler.tile_paths)]
        if self.with_gt:
            vector_tiler = tile.vector_tile.VectorTiler(dest_dir=aoi.tiles_dir_gt, verbose=True)
            vector_tiler.tile(src=aoi.gt,
                              tile_bounds=raster_tiler.tile_bounds,
                              tile_bounds_crs=raster_bounds_crs,
                              dest_fname_base=aoi.img.stem)

            updated_list = []
            for pairs_list, gt_tile_path in zip(aoi.tiles_pairs_list, sorted(vector_tiler.tile_paths)):
                updated_list.append((pairs_list[0], gt_tile_path))
            aoi.tiles_pairs_list = updated_list

    @staticmethod
    def filter_gdf(gdf: gpd.GeoDataFrame, aoi: AOI):
        """
        Filter features from a geopandas.GeoDataFrame according to an attribute field and filtering values
        @param gdf: gpd.GeoDataFrame to filter feature from
        @param aoi: an instance of AOI
        @return: Subset of source GeoDataFrame with only filtered features (deep copy)
        """
        logging.debug(gdf.columns)
        if not aoi.attr_field or not aoi.attr_vals:
            return gdf
        if not aoi.attr_field in gdf.columns:
            aoi.attr_field = aoi.attr_field.split('/')[-1]
        try:
            condList = [gdf[f'{aoi.attr_field}'] == val for val in aoi.attr_vals]
            condList.extend([gdf[f'{aoi.attr_field}'] == str(val) for val in aoi.attr_vals])
            allcond = functools.reduce(lambda x, y: x | y, condList)  # combine all conditions with OR
            gdf_filtered = gdf[allcond].copy(deep=True)
            return gdf_filtered
        except KeyError as e:
            logging.error(f'Column "{aoi.attr_field}" not found in label file {gdf.info()}')
            return gdf


def main(params):
    """
    Training and validation datasets preparation.

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
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    start_time = time.time()

    # MANDATORY PARAMETERS
    num_bands = get_key_def('number_of_bands', params['global'], expected_type=int)
    csv_file = get_key_def('prep_csv_file', params['sample'], expected_type=str)

    # OPTIONAL PARAMETERS

    # mlflow logging
    mlflow_uri = get_key_def('mlflow_uri', params['global'], default="./mlruns")
    experiment_name = get_key_def('mlflow_experiment_name', params['global'], default=f'{Path(csv_file).stem}',
                                  expected_type=str)

    # basics
    debug = get_key_def('debug_mode', params['global'], False)
    task = get_key_def('task', params['global'], 'segmentation', expected_type=str)
    if task == 'classification':
        raise ValueError(f"Got task {task}. Expected 'segmentation'.")
    elif not task == 'segmentation':
        raise ValueError(f"images_to_samples.py isn't necessary for classification tasks")
    val_percent = get_key_def('val_percent', params['sample'], default=10, expected_type=int)
    bands_idxs = get_key_def('bands_idxs', params['global'], default=None, expected_type=List)
    resize = get_key_def('resize', params['sample'], default=1)
    parallel = get_key_def('parallelize_tiling', params['sample'], default=False, expected_type=bool)

    # parameters to set output tiles directory
    data_path = Path(get_key_def('data_path', params['global'], f'./data', expected_type=str))
    Path.mkdir(data_path, exist_ok=True, parents=True)
    samples_size = get_key_def("samples_size", params["global"], default=1024, expected_type=int)
    if 'sampling_method' not in params['sample'].keys():
        params['sample']['sampling_method'] = {}
    min_annot_perc = get_key_def('min_annotated_percent', params['sample']['sampling_method'], default=0,
                                 expected_type=int)
    # TODO: use this to filter out imagery tiles full of no data
    min_raster_tile_size = get_key_def('min_raster_tile_size', params['sample'], default=0, expected_type=int)
    if not data_path.is_dir():
        raise FileNotFoundError(f'Could not locate data path {data_path}')
    attr_vals = get_key_def('target_ids', params['sample'], None, expected_type=List)

    # add git hash from current commit to parameters if available. Parameters will be saved to hdf5s
    params['global']['git_hash'] = get_git_hash()

    exp_dir = data_path / experiment_name
    if exp_dir.is_dir():
        print(f'WARNING: Data path exists: {exp_dir}. Make sure samples belong to the same experiment.')
    Path.mkdir(exp_dir, exist_ok=True, parents=True)

    # See: https://docs.python.org/2.4/lib/logging-config-fileformat.html
    console_level_logging = 'INFO' if not debug else 'DEBUG'
    set_logging(console_level=console_level_logging, logfiles_dir=exp_dir, logfiles_prefix=f'data_to_tiles_{now}')

    if debug:
        logging.warning(f'Debug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    tiler = Tiler(experiment_root_dir=exp_dir,
                  tile_size=samples_size,
                  resizing_factor=resize,
                  min_annot_perc=min_annot_perc,
                  num_bands=num_bands,
                  bands_idxs=bands_idxs)
    tiler.src_data_list = tiler.aois_from_csv(csv_path=csv_file, attr_vals=attr_vals)
    tiler.with_gt_checker()

    # VALIDATION: Assert number of bands in imagery is {num_bands}
    for aoi in tqdm(tiler.src_data_list, desc=f'Asserting number of bands in imagery is {tiler.num_bands}'):
        validate_num_bands(raster=aoi.img, num_bands=tiler.num_bands, bands_idxs=tiler.bands_idxs)

    datasets = ['trn', 'val', 'tst']

    # For each row in csv: (1) burn vector file to raster, (2) read input raster image, (3) prepare samples
    input_args = []
    logging.info(f"Preparing samples \n\tSamples_size: {samples_size} ")
    for aoi in tqdm(tiler.src_data_list, position=0, leave=False):
        try:
            # TODO: does output dir change whether GT is present or not?
            out_img_dir = aoi.tiles_dir / 'sat_img'
            out_gt_dir = aoi.tiles_dir / 'map_img' if tiler.with_gt else None
            do_tile = True
            act_img_tiles, act_gt_tiles, exp_tiles = tiler.tiling_checker(aoi.img, out_img_dir, out_gt_dir)
            if act_img_tiles == exp_tiles == act_gt_tiles or not tiler.with_gt and act_img_tiles == exp_tiles:
                logging.info('All tiles exist. Skipping tiling.\n')
                do_tile = False
            elif act_img_tiles > exp_tiles and act_gt_tiles > exp_tiles or \
                    not tiler.with_gt and act_img_tiles > exp_tiles:
                logging.error(f'\nToo many tiles for "{aoi.img}". \n'
                                 f'Expected: {exp_tiles}\n'
                                 f'Actual image tiles: {act_img_tiles}\n'
                                 f'Actual label tiles: {act_gt_tiles}\n'
                                 f'Skipping tiling.')
                do_tile = False
            elif act_img_tiles > 0 or act_gt_tiles > 0:
                logging.error(f'Missing tiles for {aoi.img}. \n'
                                 f'Expected: {exp_tiles}\n'
                                 f'Actual image tiles: {act_img_tiles}\n'
                                 f'Actual label tiles: {act_gt_tiles}\n'
                                 f'Starting tiling from scratch...')
            else:
                logging.debug(f'Expected: {exp_tiles}\n'
                              f'Actual image tiles: {act_img_tiles}\n'
                              f'Actual label tiles: {act_gt_tiles}\n'
                              f'Starting tiling from scratch...')

            # if no previous step has shown existence of all tiles, then go on and tile.
            if do_tile:
                if parallel:
                    input_args.append([tiler.tiling_per_aoi, aoi])
                else:
                    # FIXME: return img/gt tile pair for each source img/gt pair?
                    tiler.tiling_per_aoi(aoi)

        except OSError:
            logging.exception(f'An error occurred while preparing samples with "{aoi.img.stem}" (tiff) and '
                              f'{aoi.gt.stem} (gpkg).')
            continue

    if parallel:
        logging.info(f'Will tile {len(input_args)} images and labels...')
        with multiprocessing.get_context('spawn').Pool(None) as pool:
            pool.map(map_wrapper, input_args)

    logging.info(f"Tiling done. Creating pixel masks from clipped geojsons...\n"
                 f"Validation set: {val_percent} % of created training tiles")
    dataset_files = {dataset: tiler.tiles_root_dir / f'{experiment_name}_{dataset}.txt' for dataset in datasets}
    for file in dataset_files.values():
        if file.is_file():
            logging.critical(f'Dataset list exists and will be overwritten: {file}')
            file.unlink()

    datasets_kept = {dataset: 0 for dataset in datasets}
    datasets_total = {dataset: 0 for dataset in datasets}
    # TODO: clean up the second loop using tielr and aoi objects
    # loop through line of csv again
    for aoi in tqdm(tiler.src_data_list, position=0, desc='Filtering tiles and writing list to dataset text files'):
        # TODO: replace with aoi.tiles_pairs_list (check FIXMEs first)
        imgs_tiled = sorted(list(aoi.tiles_dir_img.glob('*.tif')))
        if debug:
            for sat_img_tile in tqdm(imgs_tiled, desc='DEBUG: Checking if data tiles are valid'):
                is_valid, _ = validate_raster(sat_img_tile)
                if not is_valid:
                    logging.error(f'Invalid imagery tile: {sat_img_tile}')
                map_img_tile = tiler.find_gt_tile_match(sat_img_tile, aoi.tiles_dir_gt)
                try:
                    gpd.read_file(map_img_tile)
                except Exception as e:
                    logging.error(f'Invalid ground truth tile: {sat_img_tile}. Error: {e}')
        if len(imgs_tiled) > 0 and not tiler.with_gt:
            logging.warning('List of training tiles contains no ground truth, only imagery.')
            for sat_img_tile in imgs_tiled:
                sat_size = sat_img_tile.stat().st_size
                if sat_size < min_raster_tile_size:
                    logging.debug(f'File {sat_img_tile} below minimum size ({min_raster_tile_size}): {sat_size}')
                    continue
                dataset = sat_img_tile.parts[-4]
                with open(dataset_files[dataset], 'a') as dataset_file:
                    dataset_file.write(f'{sat_img_tile.absolute()}\n')
        elif not len(imgs_tiled) == len(gts_tiled):
            msg = f"Number of imagery tiles ({len(imgs_tiled)}) and label tiles ({len(gts_tiled)}) don't match"
            logging.error(msg)
            raise IOError(msg)
        else:
            for sat_img_tile, map_img_tile in zip(imgs_tiled, gts_tiled):
                sat_size = sat_img_tile.stat().st_size
                if sat_size < min_raster_tile_size:
                    logging.debug(f'File {sat_img_tile} below minimum size ({min_raster_tile_size}): {sat_size}')
                    continue
                dataset = sat_img_tile.parts[-4]
                attr_field = info['attribute_name']
                out_px_mask = map_img_tile.parent / f'{map_img_tile.stem}.tif'
                logging.debug(map_img_tile)
                gdf = gpd.read_file(map_img_tile)
                burn_field = None
                gdf_filtered = filter_gdf(gdf, attr_field, attr_vals)

                sat_tile_fh = rasterio.open(sat_img_tile)
                sat_tile_ext = abs(sat_tile_fh.bounds.right - sat_tile_fh.bounds.left) * \
                               abs(sat_tile_fh.bounds.top - sat_tile_fh.bounds.bottom)
                annot_ct_vec = gdf_filtered.area.sum()
                annot_perc = annot_ct_vec / sat_tile_ext
                if dataset in ['trn', 'train']:
                    if annot_perc * 100 >= min_annot_perc:
                        random_val = np.random.randint(1, 100)
                        dataset = 'val' if random_val < val_percent else dataset
                        vector.mask.footprint_mask(df=gdf_filtered, out_file=str(out_px_mask),
                                                       reference_im=str(sat_img_tile),
                                                       burn_field=burn_field)
                        with open(dataset_files[dataset], 'a') as dataset_file:
                            dataset_file.write(f'{sat_img_tile.absolute()} {out_px_mask.absolute()} '
                                               f'{int(annot_perc * 100)}\n')
                        datasets_kept[dataset] += 1
                    datasets_total[dataset] += 1
                elif dataset in ['tst', 'test']:
                    vector.mask.footprint_mask(df=gdf_filtered, out_file=str(out_px_mask),
                                                   reference_im=str(sat_img_tile),
                                                   burn_field=burn_field)
                    with open(dataset_files[dataset], 'a') as dataset_file:
                        dataset_file.write(f'{sat_img_tile.absolute()} {out_px_mask.absolute()} '
                                           f'{int(annot_perc * 100)}\n')
                    datasets_kept[dataset] += 1
                    datasets_total[dataset] += 1
                else:
                    logging.error(f"Invalid dataset value {dataset} for {sat_img_tile}")

    for dataset in datasets:
        if dataset == 'train':
            logging.info(f"\nDataset: {dataset}\n"
                         f"Number of tiles with non-zero values above {min_annot_perc}%: \n"
                         f"\t Train set: {datasets_kept[dataset]}\n"
                         f"\t Validation set: {datasets_kept['val']}\n"
                         f"Number of total tiles created: {datasets_total[dataset]}\n")
        elif dataset == 'test':
            logging.info(f"\nDataset: {dataset}\n"
                         f"Number of total tiles created: {datasets_total[dataset]}\n")
    logging.info(f"End of process. Elapsed time: {int(time.time() - start_time)} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    input_type = parser.add_mutually_exclusive_group(required=True)
    input_type.add_argument('-c', '--csv', metavar='csv_file', help='Path to csv containing listed geodata with columns'
                                                                    ' as expected by geo-deep-learning. See README')
    input_type.add_argument('-p', '--param', metavar='yaml_file', help='Path to parameters stored in yaml')
    # FIXME: use hydra to better function if yaml is also used.
    parser.add_argument('--resize', default=1)
    parser.add_argument('--bands', default=None)
    # FIXME: enable BooleanOptionalAction only when GDL has moved to Python 3.8
    parser.add_argument('--debug', metavar='debug_mode', #action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--parallel', metavar='multiprocessing', #action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Boolean. If activated, will use python's multiprocessing package to parallelize")
    args = parser.parse_args()
    if args.param:
        params = read_parameters(args.param)
    elif args.csv:
        data_list = read_csv(args.csv)
        params = OrderedDict()
        params['global'] = OrderedDict()
        params['global']['debug_mode'] = args.debug
        bands_per_imagery = []
        classes_per_gt_file = []
        for data in data_list:
            with rasterio.open(data['tif'], 'r') as rdataset:
                _, metadata = validate_raster(data['tif'])
                bands_per_imagery.append(metadata['count'])
        if len(set(bands_per_imagery)) == 1:
            params['global']['number_of_bands'] = int(list(set(bands_per_imagery))[0])
            print(f"Inputted imagery contains {params['global']['number_of_bands']} bands")
        else:
            raise ValueError(f'Not all imagery has identical number of bands: {bands_per_imagery}')
        for data in data_list:
            if data['gpkg']:
                attr_field = data['attribute_name'].split('/')[-1]
                gdf = gpd.read_file(data['gpkg'])
                classes_per_gt_file.append(len(set(gdf[f'{attr_field}'])))
                print(f'Number of classes in ground truth files for attribute {attr_field}:'
                      f'\n{classes_per_gt_file}\n'
                      f'Min: {min(classes_per_gt_file)}\n'
                      f'Max: {max(classes_per_gt_file)}\n'
                      f'Number of classes will be set to max value.')
        params['global']['num_classes'] = max(classes_per_gt_file) if classes_per_gt_file else None
        params['sample'] = OrderedDict()
        params['sample']['parallelize_tiling'] = args.parallel
        params['sample']['prep_csv_file'] = args.csv

        if args.resize:
            params['sample']['resize'] = args.resize
        if args.bands:
            params['global']['bands_idxs'] = args.bands

    print(f'\n\nStarting data to tiles preparation with {args}\n'
          f'These parameters may be overwritten by a yaml\n\n')
    main(params)
