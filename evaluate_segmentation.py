import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio
from mlflow import log_metrics
from shapely.geometry import Polygon
from tqdm import tqdm
import geopandas as gpd

from utils.geoutils import clip_raster_with_gpkg, vector_to_raster
from utils.metrics import ComputePixelMetrics
from utils.utils import get_key_def, list_input_images, read_modalities
from utils.logger import get_logger
from utils.verifications import validate_num_classes, assert_crs_match

logging = get_logger(__name__)


def metrics_per_tile(label_arr: np.ndarray, pred_img: np.ndarray, input_image: rasterio.DatasetReader,
                     chunk_size: int, gpkg_name: str, num_classes: int) -> gpd.GeoDataFrame:
    """
    Compute metrics for each tile processed during inference
    @param label_arr: numpy array of label
    @param pred_img: numpy array of prediction
    @param input_image: Rasterio file handle holding the (already opened) input raster
    @param chunk_size: tile size for per-tile metrics
    @param gpkg_name: name of geopackage
    @param num_classes: number of classes
    @return:
    """
    xmin, ymin, xmax, ymax = input_image.bounds  # left, bottom, right, top
    xres, yres = (abs(input_image.transform.a), abs(input_image.transform.e))
    mx = chunk_size * xres
    my = chunk_size * yres
    h, w = input_image.shape

    feature = defaultdict(list)
    cnt = 0
    for row in tqdm(range(0, h, chunk_size), position=2, leave=False):
        for col in tqdm(range(0, w, chunk_size), position=3, leave=False):
            label = label_arr[row:row + chunk_size, col:col + chunk_size]
            pred = pred_img[row:row + chunk_size, col:col + chunk_size]
            pixelMetrics = ComputePixelMetrics(label.flatten(), pred.flatten(), num_classes)
            eval = pixelMetrics.update(pixelMetrics.iou)
            feature['id_image'].append(gpkg_name)
            for c_num in range(num_classes):
                feature['L_count_' + str(c_num)].append(int(np.count_nonzero(label == c_num)))
                feature['P_count_' + str(c_num)].append(int(np.count_nonzero(pred == c_num)))
                feature['IoU_' + str(c_num)].append(eval['iou_' + str(c_num)])
            feature['mIoU'].append(eval['macro_avg_iou'])
            logging.debug(eval['macro_avg_iou'])
            x_1, y_1 = (xmin + (col * xres)), (ymax - (row * yres))
            x_2, y_2 = (xmin + ((col * xres) + mx)), y_1
            x_3, y_3 = x_2, (ymax - ((row * yres) + my))
            x_4, y_4 = x_1, y_3
            geom = Polygon([(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)])
            feature['geometry'].append(geom)
            feature['length'].append(geom.length)
            feature['pointx'].append(geom.centroid.x)
            feature['pointy'].append(geom.centroid.y)
            feature['area'].append(geom.area)
            cnt += 1
    gdf = gpd.GeoDataFrame(feature, crs=input_image.crs.to_epsg())

    return gdf


def main(params):
    """
    Computes benchmark metrics from inference and ground truth and write results to a gpkg.
    @param params:
    @return:
    """
    start_seg = time.time()
    state_dict = Path(params['inference']['state_dict_path']).resolve(strict=True)
    modalities = read_modalities(get_key_def('modalities', params['dataset'], expected_type=str))
    num_bands = len(modalities)
    working_folder = state_dict.parent.joinpath(f'inference_{num_bands}bands')
    img_dir_or_csv = get_key_def('img_dir_or_csv_file', params['inference'], default=params['general']['raw_data_csv'],
                                 expected_type=str, to_path=True, validate_path_exists=True)
    num_classes = len(get_key_def('classes_dict', params['dataset']).keys())
    single_class_mode = True if num_classes == 1 else False
    threshold = 0.5
    debug = get_key_def('debug', params, default=False, expected_type=bool)

    # benchmark (ie when gkpgs are inputted along with imagery)
    out_gpkg = get_key_def('out_benchmark_gpkg', params['inference'], default=working_folder/"benchmark.gpkg",
                           expected_type=str)
    chunk_size = get_key_def('chunk_size', params['inference'], default=512, expected_type=int)
    dontcare = get_key_def("ignore_index", params["training"], -1)
    attribute_field = get_key_def('attribute_field', params['dataset'], None, expected_type=str)
    attr_vals = get_key_def('attribute_values', params['dataset'], None, expected_type=Sequence)

    # Assert that all values are integers (ex.: to benchmark single-class model with multi-class labels)
    if attr_vals:
        for item in attr_vals:
            if not isinstance(item, int):
                raise ValueError(f'\nValue "{item}" in attribute_values is {type(item)}, expected int.')

    list_img = list_input_images(img_dir_or_csv, glob_patterns=["*.tif", "*.TIF"])

    # VALIDATION: anticipate problems with imagery and label (if provided) before entering main for loop
    valid_gpkg_set = set()
    for info in tqdm(list_img, desc='Validating ground truth'):
        if not 'gpkg' in info.keys() and not info['gpkg']:
            raise ValueError(f"No ground truth was inputted to evaluate with")
        elif not Path(info['gpkg']).is_file():
            raise FileNotFoundError(f"Couldn't locate ground truth to evaluate with.")

        if info['gpkg'] not in valid_gpkg_set:
            validate_num_classes(vector_file=info['gpkg'],
                                 num_classes=num_classes,
                                 attribute_name=attribute_field,
                                 ignore_index=dontcare,
                                 attribute_values=attr_vals)
            assert_crs_match(info['tif'], info['gpkg'])
            valid_gpkg_set.add(info['gpkg'])

    logging.info('\nSuccessfully validated label data for benchmarking')

    gdf_ = []
    gpkg_name_ = []

    for info in tqdm(list_img, desc='Evaluating from input list', position=0, leave=True):
        local_img = Path(info['tif'])
        Path.mkdir(working_folder.joinpath(local_img.parent.name), parents=True, exist_ok=True)
        inference_image = working_folder / local_img.parent.name / f"{local_img.stem}_inference.tif"
        if not inference_image.is_file():
            raise FileNotFoundError(f"Couldn't locate inference to evaluate metrics with. Make inferece has been run "
                                    f"before you run evaluate mode.")

        pred = rasterio.open(inference_image).read()[0, ...]

        local_gpkg = Path(info['gpkg'])

        logging.info(f'\nBurning label as raster: {local_gpkg}')
        with rasterio.open(local_img, 'r') as raster:
            local_img = clip_raster_with_gpkg(raster, local_gpkg)

        raster_clipped = rasterio.open(local_img, 'r')
        logging.info(f'\nReading clipped image: {raster_clipped.name}')
        inf_meta = raster_clipped.meta

        label = vector_to_raster(vector_file=local_gpkg,
                                 input_image=raster_clipped,
                                 out_shape=(inf_meta['height'], inf_meta['width']),
                                 attribute_name=attribute_field,
                                 fill=0,  # background value in rasterized vector.
                                 attribute_values=attr_vals)
        if debug:
            logging.debug(f'\nUnique values in loaded label as raster: {np.unique(label)}\n'
                          f'Shape of label as raster: {label.shape}')

        gdf = metrics_per_tile(label_arr=label, pred_img=pred, input_image=raster_clipped, chunk_size=chunk_size,
                               gpkg_name=local_gpkg.stem, num_classes=num_classes)

        gdf_.append(gdf.to_crs(4326))
        gpkg_name_.append(local_gpkg.stem)

        if 'tracker_uri' in locals():
            pixelMetrics = ComputePixelMetrics(label, pred, num_classes)
            log_metrics(pixelMetrics.update(pixelMetrics.iou))
            log_metrics(pixelMetrics.update(pixelMetrics.dice))

    if not len(gdf_) == len(gpkg_name_):
        raise logging.critical(ValueError('\nbenchmarking unable to complete'))
    all_gdf = pd.concat(gdf_)  # Concatenate all geo data frame into one geo data frame
    all_gdf.reset_index(drop=True, inplace=True)
    gdf_x = gpd.GeoDataFrame(all_gdf, crs=4326)
    gdf_x.to_file(out_gpkg, driver="GPKG", index=False)
    logging.info(f'\nSuccessfully wrote benchmark geopackage to: {out_gpkg}')

    end_seg_ = time.time() - start_seg
    logging.info('Benchmark operation completed in {:.0f}m {:.0f}s'.format(end_seg_ // 60, end_seg_ % 60))
