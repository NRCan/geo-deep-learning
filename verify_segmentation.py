import csv
import logging
import multiprocessing
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence, Union

import rasterio
from matplotlib import pyplot as plt
from omegaconf import open_dict, DictConfig
from rasterio.plot import show_hist, show
from tqdm import tqdm

from dataset.aoi import aois_from_csv, AOI
from utils.utils import get_key_def, get_git_hash, map_wrapper


def verify_per_aoi(
        aoi: AOI,
        output_report_dir: Union[str, Path],
        extended_label_stats: bool = True,
        output_raster_stats: bool = True,
        output_raster_plots: bool = True
):
    """
    Verifies a single AOI
    @param aoi:
        AOI object containing raster and label data to verify
    @param extended_label_stats:
        if True, will calculate polygon-related stats on label (mean area, mean perimeter, mean number of vertices)
    @param output_raster_stats:
        if True, will output stats on raster radiometric data
    @param output_raster_plots:
        if True, will output plots of RGB raster and histogram for all bands
    @param output_report_dir:
        Path where output report as csv should be written.
    @return:
        Returns info on AOI or error raised, if any.
    """
    try:
        if not aoi.raster:  # in case of multiprocessing
            aoi.raster = rasterio.open(aoi.raster_dest)

        # get aoi info
        logging.info(f"\nGetting data info for {aoi.aoi_id}...")
        aoi_dict = aoi.to_dict(extended=extended_label_stats)

        # Check that `num_classes` is equal to number of classes detected in the specified attribute for each GeoPackage
        if aoi.attr_field_filter:
            label_unique_classes = aoi.label_gdf_filtered[aoi.attr_field_filter].unique()
        else:
            label_unique_classes = None
        aoi_dict['label_unique_classes'] = label_unique_classes

        if output_raster_stats:
            logging.info(f"\nGetting raster stats for {aoi.aoi_id}...")
            aoi_stats = aoi.calc_raster_stats()  # creates self.raster_np
            aoi_stats_report = {}
            for cname, stats in aoi_stats.items():
                aoi_stats_report.update(
                    {f"{cname}_{stat_name}": stat_val for stat_name, stat_val in stats['statistics'].items()})
                aoi_stats_report.update({f"{cname}_buckets": stats['histogram']['buckets']})
            aoi_dict.update(aoi_stats_report)

        if output_raster_plots:
            logging.info(f"\nGenerating plots for {aoi.aoi_id}...")
            out_plot = Path(output_report_dir) / f"raster_{aoi.aoi_id}.png"
            # https://rasterio.readthedocs.io/en/latest/topics/plotting.html
            fig, (axrgb, axhist) = plt.subplots(1, 2, figsize=(14, 7))
            aoi.raster_np = aoi.raster.read() if aoi.raster_np is None else aoi.raster_np  # prevent read if in memory
            show(aoi.raster_np, ax=axrgb, transform=aoi.raster.transform)
            show_hist(
                aoi.raster_np, bins=50, lw=1.0, stacked=False, alpha=0.75,
                histtype='step', title="Histogram", ax=axhist, label=aoi.raster_bands_request)
            plt.title(aoi.aoi_id)
            plt.savefig(out_plot)
            logging.info(f"Saved plot: {out_plot}")
            plt.close()
        return aoi_dict, None
    except Exception as e:
        raise e #logging.error(e)
        return None, e


def main(cfg: DictConfig) -> None:
    """Data verification pipeline:
    1. Get AOI infos
    2. Read or calculate stats on raster's radiometric data (min, max, median, mean, std, bincount, etc.)
    3. Generate plots for raster's radiometric data
    4. Write infos and stats to output csv.
    N.B: In current implementation, this pipeline is meant to go through no matter what error is raised.
    Errors are saved as log.
    """
    # PARAMETERS
    num_classes = len(cfg.dataset.classes_dict.keys())
    bands_requested = get_key_def('bands', cfg['dataset'], default=[], expected_type=Sequence)
    csv_file = get_key_def('raw_data_csv', cfg['dataset'], to_path=True, validate_path_exists=True)
    data_dir = get_key_def('raw_data_dir', cfg['dataset'], default="data", to_path=True, validate_path_exists=True)
    download_data = get_key_def('download_data', cfg['dataset'], default=False, expected_type=bool)

    dontcare = cfg.dataset.ignore_index if cfg.dataset.ignore_index is not None else -1
    if dontcare == 0:
        raise ValueError("\nThe 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero.")
    attribute_field = get_key_def('attribute_field', cfg['dataset'], None) #, expected_type=str)
    # Assert that all items in attribute_values are integers (ex.: single-class samples from multi-class label)
    attr_vals = get_key_def('attribute_values', cfg['dataset'], None, expected_type=(Sequence, int))

    output_report_dir = get_key_def('output_report_dir', cfg['verify'], to_path=True, validate_path_exists=True)
    output_raster_stats = get_key_def('output_raster_stats', cfg['verify'], default=False, expected_type=bool)
    output_raster_plots = get_key_def('output_raster_plots', cfg['verify'], default=False, expected_type=bool)
    extended_label_stats = get_key_def('extended_label_stats', cfg['verify'], default=False, expected_type=bool)
    parallel = get_key_def('multiprocessing', cfg['verify'], default=False, expected_type=bool)
    write_dest_raster = get_key_def('write_dest_raster', cfg['verify'], default=False, expected_type=bool)

    # ADD GIT HASH FROM CURRENT COMMIT TO PARAMETERS (if available and parameters will be saved to patches).
    with open_dict(cfg):
        cfg.general.git_hash = get_git_hash()

    logging.info(f"Building list of AOIs from input csv: {csv_file}")
    list_data_prep = aois_from_csv(
        csv_path=csv_file,
        bands_requested=bands_requested,
        attr_field_filter=attribute_field,
        attr_values_filter=attr_vals,
        download_data=download_data,
        data_dir=data_dir,
        for_multiprocessing=parallel,
        write_dest_raster=write_dest_raster,
    )

    outpath_csv = output_report_dir / f"report_info_{csv_file.stem}.csv"
    outpath_csv_errors = output_report_dir / f"report_error_{csv_file.stem}.log"

    # rename latest report if any
    if outpath_csv.is_file():
        last_mod_time_suffix = datetime.fromtimestamp(outpath_csv.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
        shutil.move(outpath_csv, outpath_csv.parent / f'{outpath_csv.stem}_{last_mod_time_suffix}.csv')
    if outpath_csv_errors.is_file():
        last_mod_time_suffix = datetime.fromtimestamp(outpath_csv_errors.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
        shutil.move(outpath_csv_errors, outpath_csv_errors.parent / f'{outpath_csv_errors.stem}_{last_mod_time_suffix}.csv')

    input_args = []
    report_list = []
    errors = []
    for aoi in tqdm(list_data_prep, position=0, desc="Verifying data"):
        if parallel:
            input_args.append([verify_per_aoi, aoi, output_report_dir, extended_label_stats,
                               output_raster_stats, output_raster_plots])
        else:
            aoi_dict, error = verify_per_aoi(aoi, output_report_dir, extended_label_stats,
                                             output_raster_stats, output_raster_plots)
            report_list.append(aoi_dict)
            errors.append(error)

    if parallel:
        logging.info(f'Parallelizing verification of {len(input_args)} aois...')
        with multiprocessing.get_context('spawn').Pool(processes=None) as pool:
            lines = pool.map_async(map_wrapper, input_args).get()
        report_list.extend([aoi_dict for aoi_dict, _ in lines])
        errors.extend([error for _, error in lines])

    logging.info(f"\nWriting to csv: {outpath_csv}...")
    with open(outpath_csv, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, report_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(report_list)

    errors = [e for e in errors if e]
    if errors:
        logging.critical(f"Verification raised {len(errors)} errors:")
        errors_str = [str(e) for e in errors]
        with open(outpath_csv_errors, 'w') as output_file:
            output_file.writelines(errors_str)
        raise Exception(errors)

    logging.info(f"\nInput data verification done. See outputs in {output_report_dir}")

