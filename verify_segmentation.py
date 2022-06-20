import csv
import logging
import shutil
from datetime import datetime
from typing import Sequence

from omegaconf import open_dict, DictConfig
from tqdm import tqdm

from dataset.aoi import aois_from_csv
from utils.utils import get_key_def, get_git_hash


def main(cfg: DictConfig) -> None:
    """TODO"""
    # PARAMETERS
    num_classes = len(cfg.dataset.classes_dict.keys())
    bands_requested = get_key_def('bands', cfg['dataset'], default=None, expected_type=Sequence)
    csv_file = get_key_def('raw_data_csv', cfg['dataset'], to_path=True, validate_path_exists=True)
    data_dir = get_key_def('raw_data_dir', cfg['dataset'], default="data", to_path=True, validate_path_exists=True)
    download_data = get_key_def('download_data', cfg['dataset'], default=False, expected_type=bool)

    dontcare = cfg.dataset.ignore_index if cfg.dataset.ignore_index is not None else -1
    if dontcare == 0:
        raise ValueError("\nThe 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero.")
    attribute_field = get_key_def('attribute_field', cfg['dataset'], None) #, expected_type=str)
    # Assert that all items in attribute_values are integers (ex.: single-class samples from multi-class label)
    attr_vals = get_key_def('attribute_values', cfg['dataset'], None, expected_type=Sequence)

    output_report_dir = get_key_def('output_report_dir', cfg['verify'], to_path=True, validate_path_exists=True)
    output_raster_stats = get_key_def('output_raster_stats', cfg['verify'], default=False, expected_type=bool)

    # ADD GIT HASH FROM CURRENT COMMIT TO PARAMETERS (if available and parameters will be saved to hdf5s).
    with open_dict(cfg):
        cfg.general.git_hash = get_git_hash()

    list_data_prep = aois_from_csv(
        csv_path=csv_file,
        bands_requested=bands_requested,
        attr_field_filter=attribute_field,
        attr_values_filter=attr_vals,
        download_data=download_data,
        data_dir=data_dir,
    )

    outpath_csv = output_report_dir / "report_verify_aois.csv"

    if outpath_csv.is_file():
        last_mod_time_suffix = datetime.fromtimestamp(outpath_csv.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
        shutil.move(outpath_csv, outpath_csv.parent / f'{outpath_csv.stem}_{last_mod_time_suffix}.csv')

    report_list = []
    for aoi in tqdm(list_data_prep, position=0):
        # get aoi info
        aoi_dict = aoi.to_dict()

        # Check that `num_classes` is equal to number of classes detected in the specified attribute for each GeoPackage
        if aoi.attr_field_filter:
            label_unique_classes = aoi.label_gdf_filtered[aoi.attr_field_filter].unique()
        else:
            label_unique_classes = None
        aoi_dict['label_unique_classes'] = label_unique_classes

        if output_raster_stats:
            aoi_stats = aoi.raster_stats()
            aoi_stats_report = {}
            for cname, stats in aoi_stats.items():
                aoi_stats_report.update({f"{cname}_{k_stat}": v_stat for k_stat, v_stat in stats['statistics'].items()})
            aoi_dict.update(aoi_stats_report)

        report_list.append(aoi_dict)

    with open(outpath_csv, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, report_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(report_list)

    logging.info(f"\nInput data verification done. See outputs in {output_report_dir}")

