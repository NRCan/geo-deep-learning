import csv
import logging
from pathlib import Path
from typing import Sequence

import omegaconf
import torch
from pandas.io.common import is_url
from solaris import vector
from solaris.utils.core import _check_rasterio_im_load
from torch.hub import load_state_dict_from_url

from dataset.aoi import aois_from_csv
from utils.metrics import iou_per_obj, iou_torchmetrics
from utils.utils import get_key_def, gdl2pl_checkpoint, read_checkpoint, \
    override_model_params_from_checkpoint, get_device_ids, set_device

from inference_segmentation import main as gdl_inference
from postprocess_segmentation import main as gdl_postprocess


def main(cfg):
    logging.debug(f"\nSetting inference parameters")
    # Main params
    csv_file = get_key_def('raw_data_csv', cfg['dataset'], to_path=True, validate_path_exists=True)
    attribute_field = get_key_def('attribute_field', cfg['dataset'], None, expected_type=str)
    # Assert that all items in attribute_values are integers (ex.: single-class samples from multi-class label)
    attr_vals = get_key_def('attribute_values', cfg['dataset'], None, expected_type=Sequence)
    if attr_vals is list:
        for item in attr_vals:
            if not isinstance(item, int):
                raise logging.critical(ValueError(f'\nAttribute value "{item}" is {type(item)}, expected int.'))

    # Main params
    root = get_key_def('root_dir', cfg['inference'], default="inference", to_path=True)
    root.mkdir(exist_ok=True)
    data_dir = get_key_def('raw_data_dir', cfg['dataset'], default="data", to_path=True, validate_path_exists=True)
    models_dir = get_key_def('checkpoint_dir', cfg['inference'], default=root / 'checkpoints', to_path=True)
    models_dir.mkdir(exist_ok=True)
    checkpoint = get_key_def('state_dict_path', cfg['inference'], expected_type=str, to_path=True, validate_path_exists=True)
    download_data = get_key_def('download_data', cfg['inference'], default=False, expected_type=bool)
    min_iou = get_key_def('iou_threshold', cfg['evaluate'], default=0.5, expected_type=float)
            
    # Create yaml to use pytorch lightning model management
    if is_url(checkpoint):
        load_state_dict_from_url(url=checkpoint, map_location='cpu', model_dir=models_dir)
        checkpoint = models_dir / Path(checkpoint).name
    checkpoint = gdl2pl_checkpoint(in_pth_path=checkpoint, out_dir=models_dir)
    checkpoint_dict = read_checkpoint(checkpoint, out_dir=models_dir)
    cfg_overridden = override_model_params_from_checkpoint(params=cfg.copy(), checkpoint_params=checkpoint_dict['params'])

    # Dataset params
    bands_requested = get_key_def('bands', cfg_overridden['dataset'], default=("red", "green", "blue"), expected_type=Sequence)

    # Set device for raster metrics
    num_devices = get_key_def('num_gpus', cfg['training'], default=1)
    gpu_devices_dict = get_device_ids(num_devices)
    device = set_device(gpu_devices_dict=gpu_devices_dict)

    # read input data from csv
    benchmark_raw_data = aois_from_csv(
        csv_path=csv_file,
        bands_requested=bands_requested,
        attr_field_filter=attribute_field,
        attr_values_filter=attr_vals,
        download_data=download_data,
        data_dir=data_dir,
    )

    metrics = []
    for aoi in benchmark_raw_data:
        metric_per_aoi = {'state_dict': Path(checkpoint).name}
        logging.info(f"Benchmarking: {aoi.aoi_id}")
        # inference on each raster
        cfg['inference']['input_stac_item'] = aoi.raster_raw_input
        cfg['inference']['output_name'] = aoi.aoi_id + '_BUIL'

        pred_raster, pred_raster_path = gdl_inference(cfg.copy())
        pred_raster = _check_rasterio_im_load(pred_raster_path).read()

        # burn to raster
        burn_val = 1 if not aoi.attr_field_filter and not aoi.attr_values_filter else None
        label_raster = vector.mask.footprint_mask(
            df=aoi.label_gdf,
            reference_im=aoi.raster,
            burn_field=aoi.attr_field_filter,
            burn_value=burn_val)

        # compute raster metrics from prediction and ground truth
        # FIXME num_classes and ignore_index
        iou = iou_torchmetrics(torch.from_numpy(pred_raster), torch.from_numpy(label_raster), device=device)
        metric_per_aoi['iou_raster'] = iou

        # compute vector metrics from prediction and ground truth
        pred_vector_path = gdl_postprocess(cfg.copy())
        #pred_vector_path = "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03_pred.gpkg"

        metric_vector = iou_per_obj(
            pred=pred_vector_path,
            gt=aoi.label,
            attr_field=aoi.attr_field_filter,
            attr_vals=[cfg['dataset']['classes_dict']['BUIL']],
            aoi_id=aoi.aoi_id,
            aoi_categ=None,  # TODO: add category for human-readable report
            min_iou=min_iou,
            gt_clip_bounds=None)
        metric_vector['iou_threshold'] = min_iou
        metric_per_aoi.update(metric_vector)
        metrics.append(metric_per_aoi)
    keys = metrics[0].keys()
    outpath = root / "benchmark.csv"
    logging.info(f"Benchmark report will be saved: ")
    if not outpath.is_file():
        with open(outpath, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(metrics)
    else:
        with open(outpath, 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writerows(metrics)


if __name__ == '__main__':
    # test benchmarking with spacenet data
    pred_raster_path = "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03_pred.tif"
    pred_raster = _check_rasterio_im_load(pred_raster_path).read()

    label_raster_path = "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03_gt.tif"
    label_raster = _check_rasterio_im_load(pred_raster_path).read()

    iou = iou_torchmetrics(torch.from_numpy(pred_raster), torch.from_numpy(label_raster), device='cuda:0')
    print(iou)

    csv_file = "tests/sampling/sampling_segmentation_binary-stac_ci.csv"
    checkpoint = "/media/data/ccmeo_models/pl_unet_resnet50_epoch-62-step-24066_gdl.ckpt"
    cfg = {'dataset': {'raw_data_csv': csv_file,
                       'classes_dict': {"BUIL": 1},
                       'raw_data_dir': "/home/remi/PycharmProjects/geo-deep-learning/tests/data/spacenet"},
           'inference': {'state_dict_path': checkpoint,
                         'download_data': True},
           'model': None,
           'evaluate': {'iou_threshold': 0.5},
           'training': {}}
    cfg = omegaconf.DictConfig(cfg)
    main(cfg)
