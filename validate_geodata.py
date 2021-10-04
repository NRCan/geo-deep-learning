import argparse
import csv
from datetime import datetime
import logging.config

import numpy as np

from data_to_tiles import set_logging

np.random.seed(1234)  # Set random seed for reproducibility

from pathlib import Path

from utils.verifications import validate_raster, is_gdal_readable

if __name__ == '__main__':
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    parser = argparse.ArgumentParser(description='Validate geodata')
    input_type = parser.add_mutually_exclusive_group(required=True)
    input_type.add_argument('-c', '--csv', metavar='csv_file', help='Path to csv containing listed geodata with columns'
                                                                    ' as expected by geo-deep-learning. See README')
    input_type.add_argument('-d', '--dir', metavar='directory', help='Directory where geodata will be validated. '
                                                                     'Recursive search is performed. All rasters that '
                                                                     'can be read by GDAL will be included')
    parser.add_argument('--debug', metavar='debug_mode', action=argparse.BooleanOptionalAction,
                        default=False)
    # Not yet implemented
    # parser.add_argument('--parallel', metavar='multiprocessing', action=argparse.BooleanOptionalAction,
    #                     default=False,
    #                     help="Boolean. If activated, will use python's multiprocessing package to parallelize")
    parser.add_argument('-e', '--extended', metavar='extended_check', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Boolean. If activated, will perform extended check on data. "
                             "WARNING: This will be require more ressources")
    args = parser.parse_args()

    debug = args.debug
    # parallel = args.parallel
    extended = args.extended
    csv_file = Path(args.csv[0]) if args.csv else args.csv
    dir = Path(args.dir) if args.dir else args.dir
    
    working_dir = csv_file.parent if csv_file else dir
    report_path = working_dir / f'report_{now}.csv'
    console_lvl = 'INFO' if not debug else 'DEBUG'
    set_logging(console_level=console_lvl, logfiles_dir=working_dir, logfiles_prefix='validate_geodata')

    if csv_file:
        raise logging.error(NotImplementedError)
    else:
        logging.info(f'Searching for GDAL readable rasters in {dir}...')
        rasters_to_validate = []
        non_rasters = []
        for file in dir.glob('**/*'):
            logging.debug(file)
            if is_gdal_readable(file):
                logging.debug(f'Found raster: {file}')
                rasters_to_validate.append(file)
            else:
                non_rasters.append(file)

        logging.info(f'Will validate {len(rasters_to_validate)} rasters...')
        report_lines = []
        header = ['raster_path', 'metadata', 'is_valid']
        for raster_path in rasters_to_validate:
            is_valid, meta = validate_raster(raster_path=raster_path, verbose=True, extended=True)
            line = [raster_path, meta, is_valid]
            report_lines.append(line)

        logging.info(f'Writing geodata validation report to: \n{report_path}')
        with open(report_path, 'w') as out:
            write = csv.writer(out)
            write.writerow(header)
            write.writerows(report_lines)

        logging.info('Done')

        # TODO: merge with verifications.py







































