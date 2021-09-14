import argparse
import functools
import math
import multiprocessing
from datetime import datetime
import logging
import logging.config
from typing import List, Union

import numpy as np
np.random.seed(1234)  # Set random seed for reproducibility
import rasterio
import time
import shutil

from pathlib import Path
from tqdm import tqdm
import solaris as sol
import geopandas as gpd

from utils.utils import get_key_def, read_csv, get_git_hash
from utils.readers import read_parameters
from utils.verifications import validate_num_classes, assert_crs_match, validate_features_from_gpkg

logging.getLogger(__name__)


def validate_raster(geo_image: Union[str, Path], verbose: bool = True):
    if not geo_image:
        return False
    geo_image = Path(geo_image) if isinstance(geo_image, str) else geo_image
    try:
        with rasterio.open(geo_image, 'r') as raster:
            metadata = raster.meta
        return True, metadata
    except rasterio.errors.RasterioIOError as e:
        metadata = ''
        if verbose:
            print(e)
        return False, metadata


def tiling_checker(src_img: Union[str, Path],
                   out_tiled_dir: Union[str, Path],
                   tile_size: int = 1024,
                   tile_stride: int = None,
                   out_suffix: str = '.tif',
                   verbose: bool = True):
    """
    Checks how many tiles should be created and compares with number of tiles already written to output directory
    @param src_img: path to source image
    @param out_tiled_dir: optional, path to output directory where tiles will be created
    @param tile_size: (int) optional, size of tile. Defaults to 1024.
    @param tile_stride: (int) optional, stride to use during tiling. Defaults to tile_size.
    @param out_suffix: optional, suffix of output tiles (ex.: ".tif" or ".geojson"). Defaults to ".tif"
    @return: number of actual tiles in output directory, number of expected tiles
    """
    tile_stride = tile_size if not tile_stride else tile_stride
    metadata = rasterio.open(src_img).meta
    tiles_x = 1 + math.ceil((metadata['width'] - tile_size) / tile_stride)
    tiles_y = 1 + math.ceil((metadata['height'] - tile_size) / tile_stride)
    nb_exp_tiles = tiles_x * tiles_y
    nb_act_tiles = len(list(out_tiled_dir.glob(f'*{out_suffix}')))
    if verbose:
        logging.info(f'Number of actual tiles with suffix "{out_suffix}": {nb_act_tiles}\n'
                     f'Number of expected tiles : {nb_exp_tiles}\n')
    return nb_act_tiles, nb_exp_tiles


def map_wrapper(x):
    '''For multi-threading'''
    return x[0](*(x[1:]))


def out_tiling_dir(root, dataset, aoi_name, category):
    root = Path(root)
    return root / dataset / aoi_name / category


def tiling(src_img: Union[str, Path],
           out_img_dir: Union[str, Path],
           tile_size: int = 1024,
           out_label_dir: Union[str, Path] = None,
           src_label: Union[str, Path] = None):
    """
    Calls solaris tiling function and outputs tiles in output directories
    @param src_img: path to source image
    @param out_img_dir: path to output tiled images directory
    @param tile_size: optional, size of tiles to output. Defaults to 1024
    @param out_label_dir: optional, path to output tiled images directory
    @param src_label: optional, path to source label (must be a geopandas compatible format like gpkg or geojson)
    @return: written tiles to output directories as .tif for imagery and .geojson for label.
    """
    raster_tiler = sol.tile.raster_tile.RasterTiler(dest_dir=out_img_dir,
                                                    src_tile_size=(tile_size, tile_size),
                                                    alpha=False,
                                                    verbose=True)
    raster_bounds_crs = raster_tiler.tile(src_img)
    if out_label_dir and src_label is not None:
        vector_tiler = sol.tile.vector_tile.VectorTiler(dest_dir=out_label_dir, verbose=True)
        vector_tiler.tile(src_label, tile_bounds=raster_tiler.tile_bounds, tile_bounds_crs=raster_bounds_crs)


def filter_gdf(gdf: gpd.GeoDataFrame, attr_field: str = None, attr_vals: List = None):
    """
    Filter features from a geopandas.GeoDataFrame according to an attribute field and filtering values
    @param gdf: gpd.GeoDataFrame to filter feature from
    @param attr_field: Name of field on which filtering operation is based
    @param attr_vals: list of integer values to keep in filtered GeoDataFrame
    @return: Subset of source GeoDataFrame with only filtered features (deep copy)
    """
    if not attr_field or not attr_vals:
        return gdf
    if not attr_field in gdf.columns:
        attr_field = attr_field.split('/')[-1]
    try:
        condList = [gdf[f'{attr_field}'] == val for val in attr_vals]
        condList.extend([gdf[f'{attr_field}'] == str(val) for val in attr_vals])
        allcond = functools.reduce(lambda x, y: x | y, condList)  # combine all conditions with OR
        gdf_filtered = gdf[allcond].copy(deep=True)
        return gdf_filtered
    except KeyError as e:
        logging.error(f'Column "{attr_field}" not found in label file {gdf.info()}')
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
    start_time = time.time()
    
    # mlflow logging
    mlflow_uri = get_key_def('mlflow_uri', params['global'], default="./mlruns")
    experiment_name = get_key_def('mlflow_experiment_name', params['global'], default='gdl-training', expected_type=str)

    # MANDATORY PARAMETERS
    num_classes = get_key_def('num_classes', params['global'], expected_type=int)
    num_bands = get_key_def('number_of_bands', params['global'], expected_type=int)
    default_csv_file = Path(get_key_def('preprocessing_path', params['global'], ''), experiment_name,
                            f"images_to_samples_{experiment_name}.csv")
    csv_file = get_key_def('prep_csv_file', params['sample'], default_csv_file, expected_type=str)

    # OPTIONAL PARAMETERS
    # basics
    debug = get_key_def('debug_mode', params['global'], False)
    task = get_key_def('task', params['global'], 'segmentation', expected_type=str)
    if task == 'classification':
        raise ValueError(f"Got task {task}. Expected 'segmentation'.")
    elif not task == 'segmentation':
        raise ValueError(f"images_to_samples.py isn't necessary for classification tasks")
    data_path = Path(get_key_def('data_path', params['global'], './data', expected_type=str))
    Path.mkdir(data_path, exist_ok=True, parents=True)
    val_percent = get_key_def('val_percent', params['sample'], default=10, expected_type=int)
    parallel = get_key_def('parallelize_tiling', params['sample'], default=True, expected_type=bool)

    # parameters to set output tiles directory
    data_path = Path(get_key_def('data_path', params['global'], './data', expected_type=str))
    samples_size = get_key_def("samples_size", params["global"], default=1024, expected_type=int)
    min_annot_perc = get_key_def('min_annotated_percent', params['sample']['sampling_method'], default=0,
                                 expected_type=int)
    if not data_path.is_dir():
        raise FileNotFoundError(f'Could not locate data path {data_path}')
    samples_folder_name = (f'tiles{samples_size}_min-annot{min_annot_perc}_{num_bands}bands'
                           f'_{experiment_name}')
    attr_vals = get_key_def('target_ids', params['sample'], None, expected_type=List)

    # add git hash from current commit to parameters if available. Parameters will be saved to hdf5s
    params['global']['git_hash'] = get_git_hash()

    final_samples_folder = None
    list_data_prep = read_csv(csv_file)

    smpls_dir = data_path / samples_folder_name
    if smpls_dir.is_dir():
        if debug:
            # Move existing data folder with a random suffix.
            last_mod_time_suffix = datetime.fromtimestamp(smpls_dir.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
            shutil.move(smpls_dir, data_path.joinpath(f'{str(smpls_dir)}_{last_mod_time_suffix}'))
        else:
            print(f'Data path exists: {smpls_dir}. Remove it or use a different experiment_name.')
    Path.mkdir(smpls_dir, exist_ok=True)

    # See: https://docs.python.org/2.4/lib/logging-config-fileformat.html
    log_config_path = Path('utils/logging.conf').absolute()
    console_level_logging = 'INFO' if not debug else 'DEBUG'
    logging.config.fileConfig(log_config_path, defaults={'logfilename': f'{smpls_dir}/{samples_folder_name}.log',
                                                         'logfilename_error':
                                                             f'{smpls_dir}/{samples_folder_name}_error.log',
                                                         'logfilename_debug':
                                                             f'{smpls_dir}/{samples_folder_name}_debug.log',
                                                         'console_level': console_level_logging})

    if debug:
        logging.warning(f'Debug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    logging.info(f'\n\tSuccessfully read csv file: {Path(csv_file).name}\n'
                 f'\tNumber of rows: {len(list_data_prep)}\n'
                 f'\tCopying first entry:\n{list_data_prep[0]}\n')

    logging.info(f'Samples will be written to {smpls_dir}\n\n')

    # Assert that all items in target_ids are integers (ex.: single-class samples from multi-class label)
    if attr_vals:
        for item in attr_vals:
            if not isinstance(item, int):
                raise ValueError(f'Target id "{item}" in target_ids is {type(item)}, expected int.')

    # VALIDATION: (1) Assert num_classes parameters == num actual classes in gpkg and (2) check CRS match (tif and gpkg)
    valid_gpkg_set = set()
    for info in tqdm(list_data_prep, position=0):
        _, metadata = validate_raster(info['tif'])
        if metadata['count'] != num_bands:
            raise ValueError(f'Imagery contains {metadata["count"]} bands, expected {num_bands}')
        if info['gpkg'] not in valid_gpkg_set:
            gpkg_classes = validate_num_classes(info['gpkg'], num_classes, info['attribute_name'], target_ids=attr_vals)
            assert_crs_match(info['tif'], info['gpkg'])
            valid_gpkg_set.add(info['gpkg'])
        if not info['dataset'] in ['trn', 'tst']:
            raise ValueError(f'Dataset value must be "trn" or "tst". Got: {info["dataset"]}')

    if debug:
        # VALIDATION (debug only): Checking validity of features in vector files
        for info in tqdm(list_data_prep, position=0, desc=f"Checking validity of features in vector files"):
            # TODO: make unit to test this with invalid features.
            invalid_features = validate_features_from_gpkg(info['gpkg'], info['attribute_name'])
            if invalid_features:
                logging.critical(f"{info['gpkg']}: Invalid geometry object(s) '{invalid_features}'")

    datasets = ['trn', 'val', 'tst']

    # For each row in csv: (1) burn vector file to raster, (2) read input raster image, (3) prepare samples
    input_args = []
    logging.info(f"Preparing samples \n\tSamples_size: {samples_size} ")
    for info in tqdm(list_data_prep, position=0, leave=False):
        try:
            out_img_dir = out_tiling_dir(smpls_dir, info['dataset'], Path(info['tif']).stem, 'sat_img')
            out_gt_dir = out_tiling_dir(smpls_dir, info['dataset'], Path(info['tif']).stem, 'map_img')

            do_tile = True
            act_img_tiles, exp_tiles = tiling_checker(info['tif'], out_img_dir,
                                                      tile_size=samples_size, out_suffix=('.tif'))
            act_gt_tiles, _ = tiling_checker(info['tif'], out_gt_dir,
                                             tile_size=samples_size, out_suffix=('.geojson'))
            if act_img_tiles == act_gt_tiles == exp_tiles:
                logging.info('All tiles exist. Skipping tiling.\n')
                do_tile = False
            elif act_img_tiles > exp_tiles and act_gt_tiles > exp_tiles:
                logging.critical(f'\nToo many tiles for "{info["tif"]}". \n'
                                 f'Expected: {exp_tiles}\n'
                                 f'Actual image tiles: {act_img_tiles}\n'
                                 f'Actual label tiles: {act_gt_tiles}\n'
                                 f'Skipping tiling.')
                do_tile = False
            elif act_img_tiles > 0 or act_gt_tiles > 0:
                logging.critical('Missing tiles for {info["tif"]}. \n'
                                 f'Expected: {exp_tiles}\n'
                                 f'Actual image tiles: {act_img_tiles}\n'
                                 f'Actual label tiles: {act_gt_tiles}\n'
                                 f'Starting tiling from scratch...')
            if do_tile:
                if parallel:
                    input_args.append([tiling, info['tif'], out_img_dir, samples_size, out_gt_dir, info['gpkg']])
                else:
                    tiling(info['tif'], out_img_dir, samples_size, out_gt_dir, info['gpkg'])

        except OSError:
            logging.exception(f'An error occurred while preparing samples with "{Path(info["tif"]).stem}" (tiff) and '
                              f'{Path(info["gpkg"]).stem} (gpkg).')
            continue

    if parallel:
        logging.info(f'Will tile {len(input_args)} images and labels')
        with multiprocessing.Pool(None) as pool:
            pool.map(map_wrapper, input_args)

    logging.info(f"Creating pixel masks from clipped geojsons\n"
                 f"Validation set: {val_percent} % of created training tiles")
    dataset_files = {dataset: smpls_dir/f'{dataset}.txt' for dataset in datasets}
    for file in dataset_files.values():
        if file.is_file():
            logging.critical(f'Dataset list exists and will be overwritten: {file}')
            file.unlink()

    datasets_kept = {dataset: 0 for dataset in datasets}
    datasets_total = {dataset: 0 for dataset in datasets}
    for info in tqdm(list_data_prep, position=0):
        out_img_dir = out_tiling_dir(smpls_dir, info['dataset'], Path(info['tif']).stem, 'sat_img')
        out_gt_dir = out_tiling_dir(smpls_dir, info['dataset'], Path(info['tif']).stem, 'map_img')
        imgs_tiled = sorted(list(out_img_dir.glob('*.tif')))
        gts_tiled = sorted(list(out_gt_dir.glob('*.geojson')))
        if not len(imgs_tiled) == len(gts_tiled):
            msg = f"Number of imagery tiles ({len(imgs_tiled)}) and label tiles ({len(gts_tiled)}) don't match"
            logging.error(msg)
            raise IOError(msg)

        for sat_img_tile, map_img_tile in zip(imgs_tiled, gts_tiled):
            dataset = sat_img_tile.parts[-4]
            attr_field = info['attribute_name'].split('/')[-1]
            out_px_mask = map_img_tile.parent / f'{map_img_tile.stem}.tif'
            gdf = gpd.read_file(map_img_tile)
            burn_field = None
            gdf_filtered = filter_gdf(gdf, attr_field, attr_vals)

            sat_tile_fh = rasterio.open(sat_img_tile)
            sat_tile_ext = abs(sat_tile_fh.bounds.right - sat_tile_fh.bounds.left) * \
                           abs(sat_tile_fh.bounds.top - sat_tile_fh.bounds.bottom)
            annot_ct_vec = gdf_filtered.area.sum()
            annot_perc = annot_ct_vec / sat_tile_ext
            if dataset in ['trn', 'train']:
                if annot_perc*100 >= min_annot_perc:
                    random_val = np.random.randint(1, 100)
                    dataset = 'val' if random_val < val_percent else dataset
                    sol.vector.mask.footprint_mask(df=gdf_filtered, out_file=str(out_px_mask),
                                                   reference_im=str(sat_img_tile),
                                                   burn_field=burn_field)
                    with open(dataset_files[dataset], 'a') as dataset_file:
                        dataset_file.write(f'{sat_img_tile} {out_px_mask} {int(annot_perc*100)}\n')
                    datasets_kept[dataset] += 1
                datasets_total[dataset] += 1
            elif dataset in ['tst', 'test']:
                sol.vector.mask.footprint_mask(df=gdf_filtered, out_file=str(out_px_mask),
                                               reference_im=str(sat_img_tile),
                                               burn_field=burn_field)
                with open(dataset_files[dataset], 'a') as dataset_file:
                    dataset_file.write(f'{sat_img_tile} {out_px_mask} {int(annot_perc*100)}\n')
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
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.ParamFile)
    print(f'\n\nStarting images to samples preparation with {args.ParamFile}\n\n')
    main(params)
