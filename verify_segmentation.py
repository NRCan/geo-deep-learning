import csv
import logging
import multiprocessing
import numbers
import shutil
from datetime import datetime
import random
from typing import Sequence

import numpy as np
import torch
from kornia.enhance import equalize_clahe
from matplotlib import pyplot as plt
from omegaconf import open_dict, DictConfig
from rasterio.plot import show_hist, show, reshape_as_image, reshape_as_raster
from skimage.exposure import exposure
from tqdm import tqdm

from dataset.aoi import aois_from_csv, AOI
from utils.utils import get_key_def, get_git_hash, map_wrapper


class RadiometricTrim(object):
    """Trims values left and right of the raster's histogram. Also called linear scaling or enhancement.
    Percentile, chosen randomly based on inputted range, applies to both left and right sides of the histogram.
    Ex.: Values below the 1.7th and above the 98.3th percentile will be trimmed if random value is 1.7"""
    def __init__(self, random_range):
        """
        :param random_range: numbers.Number (float or int) or Sequence (list or tuple) with length of 2
        """
        random_range = self.input_checker(random_range)
        self.range = random_range

    @staticmethod
    def input_checker(input_param):
        if not isinstance(input_param, (numbers.Number, Sequence)):
            raise TypeError('Got inappropriate range arg')

        if isinstance(input_param, Sequence) and len(input_param) != 2:
            raise ValueError(f"Range must be an int or a 2 element tuple or list, "
                             f"not a {len(input_param)} element {type(input_param)}.")

        if isinstance(input_param, numbers.Number):
            input_param = [input_param, input_param]
        return input_param

    def __call__(self, sample):
        # Choose trimming percentile withing inputted range
        trim = round(random.uniform(self.range[0], self.range[-1]), 1)
        # Create empty array with shape of input image
        rescaled_sat_img = np.empty(sample['sat_img'].shape, dtype=sample['sat_img'].dtype)
        # Loop through bands
        for band_idx in range(sample['sat_img'].shape[2]):
            band = sample['sat_img'][:, :, band_idx]
            band_histogram = np.bincount(sample['sat_img'].flatten())
            # Determine what is the index of nonzero pixel corresponding to left and right trim percentile
            sum_nonzero_pix_per_band = sum(band_histogram)
            left_pixel_idx = round(sum_nonzero_pix_per_band / 100 * trim)
            right_pixel_idx = round(sum_nonzero_pix_per_band / 100 * (100-trim))
            cumulative_pixel_count = 0
            # TODO: can this for loop be optimized? Also, this hasn't been tested with non 8-bit data. Should be fine though.
            # Loop through pixel values of given histogram
            for pixel_val, count_per_pix_val in enumerate(band_histogram):
                lower_limit = cumulative_pixel_count
                upper_limit = cumulative_pixel_count + count_per_pix_val
                # Check if left and right pixel indices are contained in current lower and upper pixels count limits
                if lower_limit <= left_pixel_idx <= upper_limit:
                    left_pix_val = pixel_val
                if lower_limit <= right_pixel_idx <= upper_limit:
                    right_pix_val = pixel_val
                cumulative_pixel_count += count_per_pix_val
            # Enhance using above left and right pixel values as in_range
            logging.info(f"Passing it to scikit")
            rescaled_band = exposure.rescale_intensity(band, in_range=(left_pix_val, right_pix_val), out_range='uint8')
            # Write each enhanced band to empty array
            rescaled_sat_img[:, :, band_idx] = rescaled_band
        sample['sat_img'] = rescaled_sat_img
        return sample


class CLAHE(object):
    def __init__(self, clahe):
        self.clahe = clahe

    def __call__(self, sample):
        if self.clahe:
            sat_image = torch.from_numpy(sample['sat_img'])
            sat_image = equalize_clahe(sat_image, clip_limit=5.0, grid_size=(8, 8))
            sample['sat_img'] = sat_image.numpy()
            return sample
        else:
            return sample


def verify_per_aoi(aoi: AOI, extended_label_stats, output_raster_stats, output_raster_plots, output_report_dir):
    try:
        if not aoi.raster:  # in case of multiprocessing
            aoi.raster_to_multiband()

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
            aoi_stats = aoi.calc_raster_stats()
            aoi_stats_report = {}
            for cname, stats in aoi_stats.items():
                aoi_stats_report.update(
                    {f"{cname}_{stat_name}": stat_val for stat_name, stat_val in stats['statistics'].items()})
                aoi_stats_report.update({f"{cname}_buckets": stats['histogram']['buckets']})
            aoi_dict.update(aoi_stats_report)

        if output_raster_plots:
            # logging.info(f"\nGenerating plots for {aoi.aoi_id}... STANDARD ENHANCEMENT")
            # out_plot = output_report_dir / f"raster_{aoi.aoi_id}_enh_stand_05.png"
            # # https://rasterio.readthedocs.io/en/latest/topics/plotting.html
            # fig, (axrgb, axhist) = plt.subplots(1, 2, figsize=(14, 7))
            # aoi.raster_np = aoi.raster.read() if aoi.raster_np is None else aoi.raster_np  # prevent read if in memory
            # # custom enhancements here
            # enhance_standard = RadiometricTrim(random_range=0.5)
            # sample = {'sat_img': reshape_as_image(aoi.raster_np)}
            # sample = enhance_standard(sample)
            # aoi.raster_np_enh_std = reshape_as_raster(sample['sat_img'])
            # ####
            # show(aoi.raster_np_enh_std, ax=axrgb, transform=aoi.raster.transform)
            # show_hist(
            #     aoi.raster_np_enh_std, bins=50, lw=1.0, stacked=False, alpha=0.75,
            #     histtype='step', title="Histogram", ax=axhist, label=aoi.raster_bands_request)
            # plt.title(aoi.aoi_id)
            # plt.savefig(out_plot)
            # logging.info(f"Saved plot: {out_plot}")
            # plt.close()

            logging.info(f"\nGenerating plots for {aoi.aoi_id}... CLAHE ENHANCEMENT")
            out_plot = output_report_dir / f"raster_{aoi.aoi_id}_enh_clahe_clip05.png"
            # https://rasterio.readthedocs.io/en/latest/topics/plotting.html
            fig, (axrgb, axhist) = plt.subplots(1, 2, figsize=(14, 7))
            aoi.raster_np = aoi.raster.read() if aoi.raster_np is None else aoi.raster_np  # prevent read if in memory
            # custom enhancements here
            enhance_clahe = CLAHE(clahe=True)
            sample = {'sat_img': aoi.raster_np / 255}
            sample = enhance_clahe(sample)
            aoi.raster_np_enh_std = (sample['sat_img'] * 255).astype("uint8")
            ####
            show(aoi.raster_np_enh_std, ax=axrgb, transform=aoi.raster.transform)
            show_hist(
                aoi.raster_np_enh_std, bins=50, lw=1.0, stacked=False, alpha=0.75,
                histtype='step', title="Histogram", ax=axhist, label=aoi.raster_bands_request)
            plt.title(aoi.aoi_id)
            plt.savefig(out_plot)
            logging.info(f"Saved plot: {out_plot}")
            plt.close()


        return aoi_dict, None
    except FileNotFoundError as e: #Exception as e:
        logging.error(e)
        return None, e


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
    attr_vals = get_key_def('attribute_values', cfg['dataset'], None, expected_type=(Sequence, int))

    output_report_dir = get_key_def('output_report_dir', cfg['verify'], to_path=True, validate_path_exists=True)
    output_raster_stats = get_key_def('output_raster_stats', cfg['verify'], default=False, expected_type=bool)
    output_raster_plots = get_key_def('output_raster_plots', cfg['verify'], default=False, expected_type=bool)
    extended_label_stats = get_key_def('extended_label_stats', cfg['verify'], default=False, expected_type=bool)
    parallel = get_key_def('multiprocessing', cfg['verify'], default=False, expected_type=bool)

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
        for_multiprocessing=parallel,
    )

    outpath_csv = output_report_dir / f"report_info_{csv_file.stem}.csv"
    outpath_csv_errors = output_report_dir / f"report_error_{csv_file.stem}.log"

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
            input_args.append([verify_per_aoi, aoi, extended_label_stats, output_raster_stats, output_raster_plots, output_report_dir])
        else:
            aoi_dict, error = verify_per_aoi(aoi, extended_label_stats, output_raster_stats, output_raster_plots, output_report_dir)
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
        dict_writer = csv.DictWriter(output_file, report_list[0].keys(), delimiter=';')
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

