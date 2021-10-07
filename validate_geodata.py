import argparse
import csv
import multiprocessing
from datetime import datetime
import logging.config

import numpy as np
import rasterio
from tqdm import tqdm

from data_to_tiles import set_logging
from solaris_gdl.utils.core import _check_crs, _check_gdf_load
from utils.utils import read_csv, map_wrapper

np.random.seed(1234)  # Set random seed for reproducibility

from pathlib import Path

from utils.verifications import validate_raster, is_gdal_readable, assert_crs_match, validate_features_from_gpkg


def validate_geodata(aoi: dict, validate_gt = True):
    is_valid_raster, meta = validate_raster(raster_path=aoi['tif'], verbose=True, extended=True)
    raster = rasterio.open(aoi['tif'])
    line = [Path(aoi['tif']).parent.absolute(), Path(aoi['tif']).name, meta, is_valid_raster]
    if 'gpkg' in aoi.keys():
        if not validate_gt:
            gt_line = [Path(aoi['gpkg']).parent.absolute(), Path(aoi['gpkg']).name]
            line.extend(gt_line)
            logging.info(f"Already checked ground truth vector file: {Path(aoi['gpkg']).name}")
        else:
            gt = _check_gdf_load(aoi['gpkg'])
            crs_match, epsg_raster, epsg_gt = assert_crs_match(raster, aoi['gpkg'])
            fields = gt.columns
            logging.info(f"Checking validity of features in vector files. This may take time.")
            if 'attribute_name' in aoi:
                is_valid_gt, invalid_features = validate_features_from_gpkg(gt, aoi['attribute_name'])
            else:
                is_valid_gt = f'Geometry check not implement if attribute name omitted'
                invalid_features = []
                logging.error(is_valid_gt)
            gt_line = [Path(aoi['gpkg']).parent.absolute(), Path(aoi['gpkg']).name, fields, is_valid_gt,
                       invalid_features, crs_match, epsg_raster, epsg_gt]
            line.extend(gt_line)
    return line


if __name__ == '__main__':
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    parser = argparse.ArgumentParser(description='Validate geodata')
    input_type = parser.add_mutually_exclusive_group(required=True)
    input_type.add_argument('-c', '--csv', metavar='csv_file', help='Path to csv containing listed geodata with columns'
                                                                    ' as expected by geo-deep-learning. See README')
    input_type.add_argument('-d', '--dir', metavar='directory', help='Directory where geodata will be validated. '
                                                                     'Recursive search is performed. All rasters that '
                                                                     'can be read by GDAL will be included')
    # FIXME: add BooleanOptionalAction when python >=3.8
    parser.add_argument('--debug', metavar='debug_mode', #action=argparse.BooleanOptionalAction,
                        default=False)
    # Not yet implemented
    # parser.add_argument('--parallel', metavar='multiprocessing', action=argparse.BooleanOptionalAction,
    #                     default=False,
    #                     help="Boolean. If activated, will use python's multiprocessing package to parallelize")
    parser.add_argument('-e', '--extended', metavar='extended_check', #action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Boolean. If activated, will perform extended check on data. "
                             "WARNING: This will be require more ressources")
    parser.add_argument('--parallel', metavar='multiprocessing', #action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Boolean. If activated, will use python's multiprocessing package to parallelize")
    args = parser.parse_args()

    debug = args.debug
    parallel = args.parallel
    extended = args.extended
    csv_file = Path(args.csv) if args.csv else args.csv
    dir = Path(args.dir) if args.dir else args.dir

    working_dir = csv_file.parent if csv_file else dir
    report_path = working_dir / f'validate_geodata_report_{now}.csv'
    console_lvl = 'INFO' if not debug else 'DEBUG'
    set_logging(console_level=console_lvl, logfiles_dir=working_dir, logfiles_prefix='validate_geodata')

    if csv_file:
        logging.info(f'Rasters from csv will be checked\n'
                     f'Csv file: {csv_file}')
        data_list = read_csv(args.csv)
    else:
        logging.info(f'Searching for GDAL readable rasters in {dir}...')
        data_list = []
        non_rasters = []
        for file in dir.glob('**/*'):
            logging.debug(file)
            if is_gdal_readable(file):
                logging.debug(f'Found raster: {file}')
                data_list.append({'tif': {file.absolute()}})
            else:
                # for debugging purposes
                non_rasters.append(file)

    logging.info(f'Will validate {len(data_list)} rasters...')
    report_lines = []
    validated_gts = set()
    header = ['raster_root', 'raster_path', 'metadata', 'is_valid']
    input_args = []
    for i, aoi in tqdm(enumerate(data_list), desc='Checking geodata'):
        if i == 0:
            gt_header = ['gt_root', 'gt_path', 'att_fields', 'is_valid', 'invalid_feat_ids', 'raster_gt_crs_match',
                         'epsg_raster', 'epsg_gt']
            header.extend(gt_header)

        # process ground truth data only if it hasn't been processed yet
        if aoi['gpkg'] in validated_gts:
            validate_gt = False
        else:
            validate_gt = True
            validated_gts.add(aoi['gpkg'])

        if parallel:
            input_args.append([validate_geodata, aoi, validate_gt])
        else:
            line = validate_geodata(aoi, validate_gt)
            report_lines.append(line)

    if parallel:
        logging.info(f'Will validate {len(input_args)} images and labels...')
        proc = multiprocessing.cpu_count()
        with multiprocessing.get_context('spawn').Pool(processes=proc) as pool:
            lines = pool.map_async(map_wrapper, input_args).get()
            #pool.map(map_wrapper, input_args)
        report_lines.extend(lines)

    logging.info(f'Writing geodata validation report...')
    with open(report_path, 'w') as out:
        write = csv.writer(out)
        write.writerow(header)
        write.writerows(report_lines)

    logging.info(f'Done. See report and logs: {report_path.absolute()}')

    # TODO: merge with verifications.py







































