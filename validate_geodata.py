import argparse
import csv
import multiprocessing
from datetime import datetime
import logging.config

import numpy as np
import rasterio
from tqdm import tqdm

from utils.logger import set_logging
from solaris_gdl.utils.core import _check_crs, _check_gdf_load
from utils.utils import map_wrapper, get_key_def
from utils.readers import read_gdl_csv, read_parameters

np.random.seed(1234)  # Set random seed for reproducibility

from pathlib import Path

from utils.verifications import validate_raster, is_gdal_readable, assert_crs_match, validate_features_from_gpkg


def validate_geodata(aoi: dict, validate_gt = True, extended: bool = False):
    is_valid_raster, meta = validate_raster(raster_path=aoi['tif'], verbose=True, extended=extended)
    raster = rasterio.open(aoi['tif'])
    line = [Path(aoi['tif']).parent.absolute(), Path(aoi['tif']).name, meta, is_valid_raster]
    invalid_features = []
    is_valid_gt = None
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
            if 'attribute_name' in aoi and extended:
                is_valid_gt, invalid_features = validate_features_from_gpkg(gt, aoi['attribute_name'])
            elif extended:
                is_valid_gt = f'Geometry check not implement if attribute name omitted'
                logging.error(is_valid_gt)
            gt_line = [Path(aoi['gpkg']).parent.absolute(), Path(aoi['gpkg']).name, fields, is_valid_gt,
                       invalid_features, crs_match, epsg_raster, epsg_gt]
            line.extend(gt_line)
    return line


if __name__ == '__main__':
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    parser = argparse.ArgumentParser(description='Validate geodata')
    input_type = parser.add_mutually_exclusive_group(required=True)
    input_type.add_argument('-p', '--param', metavar='yaml_file', help='Path to parameters stored in yaml')
    input_type.add_argument('-c', '--csv', metavar='csv_file', help='Path to csv containing listed geodata with columns'
                                                                    ' as expected by geo-deep-learning. See README')
    input_type.add_argument('-d', '--dir', metavar='directory', help='Directory where geodata will be validated. '
                                                                     'Recursive search is performed. All rasters that '
                                                                     'can be read by GDAL will be included')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-e', '--extended', action='store_true',
                        help="If activated, will perform extended check on data. "
                             "WARNING: This will be require more ressources")
    parser.add_argument('--parallel', action='store_true',
                        help="If activated, will use python's multiprocessing package to parallelize")
    args = parser.parse_args()

    debug = args.debug
    parallel = args.parallel
    extended = args.extended
    if args.csv:
        working_dir = Path(args.csv).parent
    elif args.param:
        working_dir = Path(args.param).parent
    else:
        working_dir = Path(args.dir) if args.dir else args.dir

    report_path = working_dir / f'validate_geodata_report_{now}.csv'
    console_lvl = 'INFO' if not debug else 'DEBUG'
    set_logging(console_level=console_lvl, logfiles_dir=working_dir, logfiles_prefix='validate_geodata')

    if args.param:
        params = read_parameters(args.param)
        csv_file = get_key_def('prep_csv_file', params['sample'], expected_type=str)

        data_list = read_gdl_csv(csv_file)
        logging.info(f'Found preprocessing csv file in yaml: {csv_file}\n'
                     f'{len(data_list)} rasters from csv will be checked\n')
    elif args.csv:
        csv_file = Path(args.csv)
        logging.info(f'Rasters from csv will be checked\n'
                     f'Csv file: {csv_file}')
        data_list = read_gdl_csv(args.csv)
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
            input_args.append([validate_geodata, aoi, validate_gt, extended])
        else:
            line = validate_geodata(aoi, validate_gt, extended)
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







































