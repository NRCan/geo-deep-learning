import difflib
import math
import multiprocessing
import shutil

import matplotlib.pyplot
from typing import Union, Sequence, List
from pathlib import Path

import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box
from solaris import tile, vector
from solaris.utils.core import _check_gdf_load, _check_rasterio_im_load
from tqdm import tqdm
from omegaconf import DictConfig, open_dict

# Our modules
from dataset.aoi import aois_from_csv, AOI
from utils.utils import (
    get_key_def, get_git_hash
)
from utils.verifications import validate_raster
# Set the logging file
from utils import utils
logging = utils.get_logger(__name__)  # import logging
# Set random seed for reproducibility
np.random.seed(1234)


def annot_percent(img_tile: Union[str, Path, rasterio.DatasetReader],
                  gdf_tile: Union[str, Path, gpd.GeoDataFrame],
                  ):
    """
    Calculate percentage of values in GeoDataFrame that contain classes other than background
    @param img_tile: str, Path or rasterio.DatasetReader
    @param gdf_tile: str, Path or gpd.GeoDataFrame
    @return: (int) Annotated percent
    """
    gdf_tile = _check_gdf_load(str(gdf_tile))
    if gdf_tile.empty:
        return 0
    img_tile_dataset = _check_rasterio_im_load(str(img_tile))
    # FIXME: this may not work if bounds are reprojected during tiling
    tile_bounds = box(*img_tile_dataset.bounds)

    annot_ct_vec = gdf_tile.area.sum()
    annot_perc = annot_ct_vec / tile_bounds.area
    return annot_perc * 100


class Tiler(object):
    def __init__(self,
                 experiment_root_dir: Union[Path, str],
                 src_data_list: List = None,
                 tile_size: int = 1024,
                 tile_stride: int = None,
                 resizing_factor: Union[int, float] = 1,
                 min_annot_perc: int = 0,
                 num_bands: int = None,
                 bands_idxs: Sequence = None,
                 val_percent: int = None,
                 attr_field_exp: str = None,
                 attr_vals_exp: list = None,
                 overwrite: bool = False,
                 debug: bool = False):
        """
        @param experiment_root_dir: pathlib.Path or str
            Root directory under which all tiles will written (in subfolders)
        @param src_data_list: list
            List of objects of class AOI
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
        @param val_percent: integer, optional
            Percentage of training tiles that should be written to validation set
        @param attr_field_exp: str, optional
            Attribute field for ground truth tiles across entire experiment from which features will filtered based on
            values provided
        @param attr_vals_exp: list, optional
            Values from attribute field for which features will be kept. These values must apply across entire dataset
        @param overwrite: bool, optional
            If True, output directories will be erased prior to tiling if they exist
        """
        if src_data_list and not isinstance(src_data_list, List):
            raise TypeError('Input data should be a List')
        self.src_data_list = src_data_list
        if not isinstance(experiment_root_dir, (Path, str)):
            raise TypeError(f'Tiles root directory should be a of class pathlib.Path or a string.\n'
                            f'Got {experiment_root_dir} of type {type(experiment_root_dir)}')
        if not Path(experiment_root_dir).is_dir():
            raise FileNotFoundError(f'{experiment_root_dir} is not a valid directory')
        self.tiles_root_dir = Path(experiment_root_dir)
        self.datasets = ('trn', 'val', 'tst')  # in order, should be (train, validation, test)

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
            num_bands_set = set([aoi.raster.meta['count'] for aoi in src_data_list.values()])
            if len(num_bands_set) > 1:
                raise ValueError(f'Not all imagery has equal number of bands. '
                                 f'Check imagery or define bands indexes to keep. \n'
                                 f'Number of bands found: {num_bands_set}')
        self.num_bands = num_bands

        self.tiles_dir_name = self.make_tiles_dir_name(self.dest_tile_size, self.num_bands)

        self.tiles_root_dir = experiment_root_dir / self.tiles_dir_name

        if bands_idxs and not isinstance(bands_idxs, Sequence) and self.num_bands == len(bands_idxs):
            raise TypeError(f"Bands indexes should be a list of same length as number of bands ({self.num_bands}).\n"
                            f"Bands_idxs: {bands_idxs}\n"
                            f"num_bands: {num_bands}")
        elif bands_idxs and 0 in bands_idxs:
            raise ValueError(f'As per rasterio convention, band indexing starts at 1, not 0')
        self.bands_idxs = bands_idxs

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

        self.overwrite = overwrite
        self.debug = debug

        if self.tiles_root_dir.is_dir():
            logging.warning(f'\nTiles root directory exists: {self.tiles_root_dir}.')
            if overwrite:
                logging.warning(f'\nPrevious chips will be erased.'
                                f'\nMake sure chips belong to the same experiment.')
                shutil.rmtree(self.tiles_root_dir)
            else:
                logging.critical(f'\nTiling failed. To overwrite existing chips, set tiling.overwrite to "True"')

        Path.mkdir(self.tiles_root_dir, parents=True, exist_ok=True)
        [Path.mkdir(self.tiles_root_dir/dataset, exist_ok=True) for dataset in self.datasets]
        logging.info(f'Tiles will be written to {self.tiles_root_dir}\n\n')

    def with_gt_checker(self):
        for aoi in self.src_data_list:
            if not aoi.label:
                self.with_gt = False
                logging.warning(f"No ground truth data found for {aoi.raster.name}. Only imagery will be processed from now on")

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
        dataset_file_name = f'{exp_name}{sampling_str}_{dataset}.csv'
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
                       src_img_meta: dict,
                       dest_img_tiles_dir: Union[str, Path],
                       dest_gt_tiles_dir: Union[str, Path] = None,
                       verbose: bool = True):
        """
        Checks how many tiles should be created and compares with number of tiles already written to output directory
        @param src_img_meta: path to source image
        @param dest_img_tiles_dir: optional, path to output directory where imagery tiles will be created
        @param dest_gt_tiles_dir: optional, path to output directory where ground truth tiles will be created
        @return: number of actual tiles in output directory, number of expected tiles
        """
        act_data_tiles = []
        dest_width = src_img_meta['width'] * self.resizing_factor
        dest_height = src_img_meta['height'] * self.resizing_factor
        tiles_x = 1 + math.ceil((dest_width - self.dest_tile_size) / self.tile_stride)
        tiles_y = 1 + math.ceil((dest_height - self.dest_tile_size) / self.tile_stride)
        nb_exp_tiles = tiles_x * tiles_y
        # glob for tiles of the vector ground truth if 'geojson' is in the suffix
        act_img_tiles = list(dest_img_tiles_dir.glob(f'{Path(src_img_meta["name"]).stem}*.tif'))
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
            act_data_tiles = [img_tile for img_tile in act_img_tiles]

        if verbose:
            logging.info(f'Number of actual imagery tiles : {nb_act_img_tiles}\n'
                         f'Number of actual ground truth tiles : {nb_act_gt_tiles}\n'
                         f'Number of expected tiles : {nb_exp_tiles}\n')
        return act_data_tiles, nb_act_img_tiles, nb_act_gt_tiles, nb_exp_tiles

    def get_src_tile_size(self):
        """
        FIXME: remove!
        Sets outputs dimension of source tile if resizing, given destination size and resizing factor
        @param dest_tile_size: (int) Size of tile that is expected as output
        @param resize_factor: (float) Resize factor to apply to source imagery before outputting tiles
        @return: (int) Source tile size
        """
        if self.resizing_factor:
            return int(self.dest_tile_size / self.resizing_factor)
        else:
            return self.dest_tile_size

    def tiling_per_aoi(
            self,
            aoi: AOI,
            out_img_dir: Union[str, Path],
            out_label_dir: Union[str, Path] = None):
        """
        Calls solaris_gdl tiling function and outputs tiles in output directories

        @param out_img_dir: path to output tiled images directory
        @param out_label_dir: optional, path to output tiled images directory
        @return: written tiles to output directories as .tif for imagery and .geojson for label.
        """
        if not aoi.raster:  # in case of multiprocessing
            aoi.raster_to_multiband()

        self.src_tile_size = self.get_src_tile_size()
        raster_tiler = tile.raster_tile.RasterTiler(dest_dir=out_img_dir,
                                                    src_tile_size=(self.src_tile_size, self.src_tile_size),
                                                    dest_tile_size=(self.dest_tile_size, self.dest_tile_size),
                                                    alpha=False,
                                                    verbose=True)
        raster_bounds_crs = raster_tiler.tile(aoi.raster) #, channel_idxs=self.bands_idxs)
        logging.debug(f'Raster bounds crs: {raster_bounds_crs}\n'
                      f'')

        vec_tler_tile_paths = vec_tler_tile_bds_reprojtd = None
        if self.with_gt:
            dest_crs = raster_tiler.dest_crs if raster_tiler.dest_crs.is_epsg_code else None
            vec_tler = tile.vector_tile.VectorTiler(dest_dir=out_label_dir,
                                                    dest_crs=dest_crs,
                                                    verbose=True,
                                                    super_verbose=self.debug)
            vec_tler.tile(src=str(aoi.label),
                          tile_bounds=raster_tiler.tile_bounds,
                          tile_bounds_crs=raster_bounds_crs,
                          dest_fname_base=Path(aoi.raster_meta['name']).stem)
            vec_tler_tile_paths = vec_tler.tile_paths

        aoi.raster = None  # for multiprocessing

        return aoi, raster_tiler.tile_paths, vec_tler_tile_paths

    def filter_tile_pair(self,
                         aoi: AOI,
                         img_tile: Union[str, Path],
                         gt_tile: Union[str, Path, gpd.GeoDataFrame],
                         dataset: str,
                         ):
        map_img_gdf = _check_gdf_load(gt_tile)
        annot_perc = annot_percent(
            img_tile=img_tile,
            gdf_tile=map_img_gdf,
        )
        # Then filter by annotated percentage
        # FIXME: keeping all val chips will bust val_percent if min annot > 0, but more logical to bypass conditions for val chips
        if annot_perc >= self.min_annot_perc or dataset in self.datasets[1:]:  # val/tst datasets don't need to meet conditions
            return True, annot_perc
        elif dataset == self.datasets[0]:  # trn
            logging.debug(f"Ground truth tile in training dataset doesn't reach minimum annotated percentage.\n"
                          f"Ground truth tile: {gt_tile}\n"
                          f"Annotated percentage: {annot_perc}\n"
                          f"Minimum annotated percentage: {self.min_annot_perc}")
            return False, annot_perc
        else:
            raise ValueError(f'dataset should be "trn", "val" or "tst", got {dataset}')

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
                     ):
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
        if not aoi.attr_field_filter and aoi.attr_values_filter is not None:
            raise ValueError(f'Values for an attribute field have been provided, but no attribute field is set.\n'
                             f'Attribute values: {aoi.attr_values_filter}')
        elif aoi.attr_field_filter is not None and not aoi.attr_values_filter:
            raise ValueError(f'An attribute field has been provided, but no attribute values were set.\n'
                             f'Attribute field: {aoi.attr_field_filter}. If all values from attribute fields are '
                             f'to be kept, please input full list of values in dataset configuration.')
        # Burn value in attribute field from which features are being filtered
        burn_field = aoi.attr_field_filter if aoi.attr_field_filter else None
        # no attribute field or val given means all values should be burned to 1
        burn_val = 1 if not aoi.attr_field_filter and not aoi.attr_values_filter else None
        if gt_tile.empty:
            burn_field = None
        elif aoi.attr_field_filter:
            # Define new column 'burn_val' with continuous values for use during burning
            cont_vals_dict = {src: (dst+1 if continuous else src) for dst, src in enumerate(aoi.attr_values_filter)}
            if all(isinstance(val, str) for val in gt_tile[aoi.attr_field_filter].unique().tolist()):
                cont_vals_dict = {str(src): dst for src, dst in cont_vals_dict.items()}
            gt_tile['burn_val'] = gt_tile[aoi.attr_field_filter].map(cont_vals_dict)
            burn_field = 'burn_val'  # overwrite burn_field
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

    def filter_and_burn_dataset(
            self,
            aoi: AOI,
            img_tile: Union[str, Path],
            gt_tile: Union[str, Path],
            continuous_vals: bool = True,
            save_preview_labels: bool = False,
    ):
        """
        TODO
        @param aoi:
        @param img_tile:
        @param gt_tile:
        @param continuous_vals:
        @param save_preview_labels:
        @return:
        """
        if not aoi.raster:  # in case of multiprocessing
            aoi.raster_to_multiband()

        random_val = np.random.randint(1, 100)
        # for trn tiles, sort between trn and val based on random number
        dataset = self.datasets[1] if aoi.split == 'trn' and random_val < self.val_percent else aoi.split
        if dataset == self.datasets[1]:  # val dataset  # TODO refactor to function
            img_tile_dest_parts = [part if part != aoi.split else self.datasets[1] for part in img_tile.parts]
            gt_tile_dest_parts = [part if part != aoi.split else self.datasets[1] for part in gt_tile.parts]
            img_tile_dest, gt_tile_dest = Path(*img_tile_dest_parts), Path(*gt_tile_dest_parts)
            Path.mkdir(img_tile_dest.parent, exist_ok=True, parents=True)
            Path.mkdir(gt_tile_dest.parent, exist_ok=True, parents=True)
            shutil.move(img_tile, img_tile_dest)
            shutil.move(gt_tile, gt_tile_dest)
            img_tile, gt_tile = img_tile_dest, gt_tile_dest
        out_gt_burned_path = self.get_burn_gt_tile_path(attr_vals=aoi.attr_values_filter, gt_tile=gt_tile)
        gdf_tile = AOI.filter_gdf_by_attribute(
            gdf_tile=str(gt_tile),
            attr_field=aoi.attr_field_filter,
            attr_vals=aoi.attr_values_filter
        )
        keep_tile_pair, annot_perc = self.filter_tile_pair(
            aoi,
            img_tile=img_tile,
            gt_tile=gdf_tile,
            dataset=dataset,
        )
        logging.debug(annot_perc)
        if keep_tile_pair:
            self.burn_gt_tile(aoi,
                              img_tile=img_tile,
                              gt_tile=gdf_tile,
                              out_px_mask=out_gt_burned_path,
                              continuous=continuous_vals,
                              save_preview=save_preview_labels,
                              )
            dataset_line = f'{Path(img_tile).absolute()};{Path(out_gt_burned_path).absolute()};{round(annot_perc)}\n'
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


def map_wrapper(x):
    """For multi-threading"""
    return x[0](*(x[1:]))


def main(cfg: DictConfig) -> None:
    """
    Creates training, validation and testing datasets preparation. The tiling process consists of cutting up the imagery
    and ground truth to tiles of a certain size. This prepares the dataset for training.

    Tile size

    Size of an individual tile. For example, a raster of 1024 x 1024 pixels will output 4 tiles if tile_size is 512. The
    value for this parameter should remain relatively stable as varying tile sizes has little impact of the performance
    of model. Tiling is mostly aimed at making is possible to fill a batch with at least 4 chip pairs without busting a
    machine's memory while training. Defaults to 512.

    Minimum annotated percent

    Discards tile pairs (imagery & ground truth) if the non-background area (e.g. area covered with classes of interest)
    on a given ground truth tile is lower than this minimum. Defaults to 0 (keep all tiles). This parameter is a data
    balancing tool for undersampling. It is easy to implement and use, but may not be the perfect solution for all data
    balancing problems. For more information on pros and cons of undersampling, oversampling and other class
    balancing strategies, see [*Buda & Al., 2018*](https://www.sciencedirect.com/science/article/pii/S0893608018302107?casa_token=1gtjUgWc6pUAAAAA:SUDHxtgD8SPDrsM4wR93mH6ZYW57Mr-BYX2nBwxTuT8DsUlWJcvpAV1vgdACQgY78IbiZuCrPgb_)
    and [*Longadge & Dongre, 2013*](https://arxiv.org/pdf/1305.1707).

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
    bands_requested = get_key_def('bands', cfg['dataset'], default=None, expected_type=Sequence)
    num_bands = len(bands_requested)
    experiment_name = get_key_def('project_name', cfg['general'], default='gdl-training')
    debug = cfg.debug

    # RAW DATA PARAMETERS
    data_dir = get_key_def('raw_data_dir', cfg['dataset'], to_path=True, validate_path_exists=True)
    csv_file = get_key_def('raw_data_csv', cfg['dataset'], to_path=True, validate_path_exists=True)
    download_data = get_key_def('download_data', cfg['dataset'], default=False, expected_type=bool)
    tiles_root_dir = get_key_def('tiles_data_dir', cfg['tiling'], default=data_dir, to_path=True,
                                 validate_path_exists=True)

    # SAMPLE PARAMETERS
    samples_size = get_key_def('tile_size', cfg['tiling'], default=512, expected_type=int)
    min_annot_perc = get_key_def('min_annot_perc', cfg['tiling'], default=0)
    continuous_vals = get_key_def('continuous_values', cfg['tiling'], default=True)
    save_prev_labels = get_key_def('save_preview_labels', cfg['tiling'], default=False)
    resize = get_key_def('resampling', cfg['tiling'], default=1, expected_type=int)
    parallel = get_key_def('multiprocessing', cfg['tiling'], default=False, expected_type=bool)
    overwrite = get_key_def('overwrite', cfg['tiling'], default=False, expected_type=bool)

    val_percent = int(get_key_def('train_val_percent', cfg['dataset'], default=0.3)['val'] * 100)
    attr_field = get_key_def('attribute_field', cfg['dataset'], None, expected_type=str)
    attr_vals = get_key_def('attribute_values', cfg['dataset'], None, expected_type=(Sequence, int))

    # ADD GIT HASH FROM CURRENT COMMIT TO PARAMETERS (if available and parameters will be saved to hdf5s).
    with open_dict(cfg):
        cfg.general.git_hash = get_git_hash()

    exp_dir = tiles_root_dir / experiment_name

    if exp_dir.is_dir():
        logging.warning(f'Data path exists: {exp_dir}. Make sure samples belong to the same experiment.')
    Path.mkdir(exp_dir, exist_ok=True, parents=True)

    if debug:
        logging.warning(f'Debug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    # FIXME: KeyError with tiler.datasets if aoi.split == inference.
    tiler = Tiler(experiment_root_dir=exp_dir,
                  tile_size=samples_size,
                  resizing_factor=resize,
                  min_annot_perc=min_annot_perc,
                  num_bands=num_bands,
                  val_percent=val_percent,
                  attr_field_exp=attr_field,
                  attr_vals_exp=attr_vals,
                  overwrite=overwrite,
                  debug=debug)
    tiler.src_data_list = aois_from_csv(
        csv_path=csv_file,
        bands_requested=bands_requested,
        attr_field_filter=attr_field,
        attr_values_filter=attr_vals,
        download_data=download_data,
        data_dir=data_dir,
        for_multiprocessing=parallel,
    )
    tiler.with_gt_checker()

    # For each row in csv: (1) tiling imagery and labels
    input_args = []
    tilers = []
    logging.info(f"Preparing samples \n\tSamples_size: {samples_size} ")
    for index, aoi in tqdm(enumerate(tiler.src_data_list), position=0, leave=False):
        try:
            # FIXME
            tiles_root_name = tiler.make_tiles_dir_name(
                tile_size=samples_size,
                num_bands=num_bands)
            tiles_dir = exp_dir / tiles_root_name / aoi.split.strip() / aoi.aoi_id.strip()
            tiles_dir_img = tiles_dir / 'images'
            tiles_dir_gt = tiles_dir / 'labels' if tiler.with_gt else None

            do_tile = True
            data_pairs, nb_act_img_t, nb_act_gt_t, nb_exp_t = tiler.tiling_checker(aoi.raster_meta, tiles_dir_img, tiles_dir_gt)
            # add info about found tiles for each aoi by aoi's index
            tiler.src_data_list[index].tiles_pairs_list = [data_pair for data_pair in data_pairs]
            if nb_act_img_t == nb_exp_t == nb_act_gt_t or not tiler.with_gt and nb_act_img_t == nb_exp_t:
                logging.info('All tiles exist. Skipping tiling.\n')
                do_tile = False
            elif nb_act_img_t > nb_exp_t and nb_act_gt_t > nb_exp_t or \
                    not tiler.with_gt and nb_act_img_t > nb_exp_t:
                logging.error(f'\nToo many tiles for "{aoi.aoi_id}". \n'
                              f'Expected: {nb_exp_t}\n'
                              f'Actual image tiles: {nb_act_img_t}\n'
                              f'Actual label tiles: {nb_act_gt_t}\n'
                              f'Skipping tiling.')
                do_tile = False
            elif nb_act_img_t > 0 or nb_act_gt_t > 0:
                logging.error(f'Missing tiles for {aoi.aoi_id}. \n'
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
            if do_tile:
                # FIXME: aoi cannot be a rasterio Dataset (cannot be pickled)
                # https://github.com/rasterio/rasterio/issues/1731
                if parallel:
                    input_args.append([tiler.tiling_per_aoi, aoi, tiles_dir_img, tiles_dir_gt])
                else:
                    try:
                        tiler_pair = tiler.tiling_per_aoi(aoi, out_img_dir=tiles_dir_img, out_label_dir=tiles_dir_gt)
                        tilers.append(tiler_pair)
                    except ValueError as e:
                        logging.debug(f'Failed to tile\n'
                                      f'Img: {aoi.raster.name}\n'
                                      f'GT: {aoi.label}')
                        raise e

        except OSError:
            logging.exception(f'An error occurred while preparing samples with "{Path(aoi.raster.name).stem}" (tiff) and '
                              f'{aoi.label.stem} (gpkg).')
            continue

    if parallel:
        logging.info(f'Will tile {len(input_args)} images and labels...')
        with multiprocessing.get_context('spawn').Pool(None) as pool:
            tilers = pool.map_async(map_wrapper, input_args).get()

    for index, aoi in tqdm(
            enumerate(tiler.src_data_list),
            position=0,
            leave=False,
            desc=f"Updating AOIs' information about their tiles paths"):
        for tiled_aoi, raster_tiler_paths, vec_tler_paths in tqdm(tilers):
            if tiled_aoi == aoi:
                for zipped in zip(raster_tiler_paths, vec_tler_paths):
                    img_tile_path, gt_tile_path = zipped
                    tiler.src_data_list[index].tiles_pairs_list.append((img_tile_path, gt_tile_path))

    logging.info(f"Tiling done. Creating pixel masks from clipped geojsons...\n"
                 f"Validation set: {val_percent} % of created training tiles")
    # TODO: how does train_segmentation know where these are?
    dataset_files = {}
    for dset in tiler.datasets:
        name, _ = tiler.make_dataset_file_name(experiment_name, tiler.min_annot_perc, dset, tiler.attr_vals_exp)
        dset_path = tiler.tiles_root_dir / name
        if dset_path.is_file():
            logging.critical(f'Dataset list exists and will be overwritten: {dset_path}')
            dset_path.unlink()
        dataset_files[dset] = dset_path

    input_args = []
    dataset_lines = []
    datasets_total = {dataset: 0 for dataset in tiler.datasets}
    if not tiler.with_gt:
        logging.warning('List of training tiles contains no ground truth, only imagery.')
    # loop through line of csv again and
    # (1) filter out training data that doesn't match user-defined conditions such as minimum annotated percent
    # (2) burn filtered labels to raster format
    for aoi in tqdm(tiler.src_data_list, position=0,
                    desc='Looping in AOIs'):
        if debug:
            for img_tile, gt_tile in tqdm(aoi.tiles_pairs_list,
                                             desc='DEBUG: Checking if data tiles are valid'):
                try:
                    validate_raster(str(img_tile))
                except Exception as e:
                    logging.error(f'\nInvalid imagery tile: {img_tile}'
                                  f'\n{e}')
                try:
                    _check_gdf_load(gt_tile)  # validates ground truth tile
                except Exception as e:
                    logging.error(f'\nInvalid ground truth tile: {img_tile}. '
                                  f'\n{e}')

        for img_tile, gt_tile in tqdm(aoi.tiles_pairs_list, position=1,
                                           desc=f'Filter {len(aoi.tiles_pairs_list)} tiles and burn ground truth'):
            datasets_total[aoi.split] += 1
            # If for inference, write only image tile since there's no ground truth
            if not tiler.with_gt:
                if gt_tile is not None:
                    logging.error(f'Tiler set without ground truth, but ground truth was found. It will be ignored.\n'
                                  f'Image tile: {img_tile}\n'
                                  f'Ground truth: {gt_tile}')
                dataset_line = f'{img_tile.absolute()}\n'
                dataset_lines.append((aoi.split, dataset_line))
            # if for train, validation or test dataset, then filter, burn and provided complete line to write to file
            else:
                if parallel:
                    input_args.append([tiler.filter_and_burn_dataset, aoi, img_tile, gt_tile])
                else:
                    line_tuple = tiler.filter_and_burn_dataset(aoi,
                                                               img_tile=img_tile,
                                                               gt_tile=gt_tile,
                                                               continuous_vals=continuous_vals,
                                                               save_preview_labels=save_prev_labels)
                    dataset_lines.append(line_tuple)

    if parallel:
        logging.info(f'Parallelizing burning of {len(input_args)} filtered ground truth tiles...')
        proc = multiprocessing.cpu_count()
        with multiprocessing.get_context('spawn').Pool(processes=proc) as pool:
            lines = pool.map_async(map_wrapper, input_args).get()
        dataset_lines.extend(lines)

    # write to dataset text file the data that was kept after filtering
    datasets_kept = {dataset: 0 for dataset in tiler.datasets}
    for line_tuple in tqdm(dataset_lines, desc=f"Writing {len(dataset_lines)} lines to dataset files"):
        dataset, dataset_line = line_tuple
        if dataset_line:
            with open(dataset_files[dataset], 'a') as dataset_file:
                dataset_file.write(dataset_line)
                datasets_kept[dataset] += 1

    # final report
    logging.info(f"\nExpected val/trn ratio: {tiler.val_percent} %"
                 f"\nActual val/trn ratio: {datasets_kept['val'] / (datasets_kept['trn'] * 100 + 1e-10):.2f} %")
    for dataset in tiler.datasets:
        if dataset == 'trn':
            logging.info(f"\nDataset: {dataset}"
                         f"\n\tFilters:"
                         f"\n\t - Minimumum non-zero values (aka minimum annotated percentage): {min_annot_perc}%"
                         f"\n\tKept: {datasets_kept['trn']}"
                         f"\n\tDiscarded: {datasets_total[dataset] - datasets_kept['val'] - datasets_kept['trn']}")
        else:
            logging.info(f"\nDataset: {dataset}"
                         f"\n\tTotal tiles: {datasets_kept[dataset]}")
    logging.info(f"\nEnd of process. See tiled dataset lists:")
    for dataset, file in dataset_files.items():
        logging.info(f"{dataset}: {str(file)}")
