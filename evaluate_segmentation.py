import csv
import logging
import multiprocessing
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import omegaconf
import torch
from omegaconf import OmegaConf, open_dict
from pandas.io.common import is_url
from solaris import vector
from solaris.utils.core import _check_rasterio_im_load
from torch.hub import load_state_dict_from_url

from dataset.aoi import aois_from_csv
from utils.geoutils import create_new_raster_from_base
from utils.metrics import iou_per_obj, iou_torchmetrics
from utils.utils import get_key_def, gdl2pl_checkpoint, read_checkpoint, \
    override_model_params_from_checkpoint, get_device_ids, extension_remover, set_device

from inference_segmentation import main as gdl_inference
from postprocess_segmentation import main as gdl_postprocess


def map_wrapper(x):
    """For multi-threading"""
    return x[0](*(x[1:]))


def benchmark_per_aoi(cfg, checkpoint, root, aoi, heatmap_threshold, device, num_classes, min_iou, outpath_csv, keys):
    metric_per_aoi = OrderedDict(
        {'aoi_id': aoi.aoi_id, 'state_dict': Path(checkpoint).name, 'heatmap_threshold': heatmap_threshold})
    logging.info(f"Benchmarking: {aoi.aoi_id}")
    # inference on each raster
    cfg['inference']['input_stac_item'] = item_url = aoi.raster_raw_input
    cfg['inference']['output_name'] = aoi.aoi_id + '_BUIL'

    # inference output path
    outname = get_key_def('output_name', cfg['inference'], default=f"{Path(item_url).stem}_pred")
    outname = extension_remover(outname)
    outpath_heat = root / f"{outname}_heatmap.tif"

    # postprocess output path
    out_poly_suffix = get_key_def('polygonization', cfg['postprocess']['output_suffixes'], default='_raw',
                                  expected_type=str)
    pred_vector_path = root / f"{outname}{out_poly_suffix}.gpkg"
    out_raster = root / f"{pred_vector_path.stem}_metrics_thresh{str(heatmap_threshold)}.tif"
    out_vector = root / f"{pred_vector_path.stem}_metrics_iou{str(min_iou).replace('.', '')}_thresh{str(heatmap_threshold)}.gpkg"

    with open_dict(cfg):
        OmegaConf.update(cfg, 'inference.output_name', str(out_raster.stem), merge=False)
        OmegaConf.update(cfg, 'inference.heatmap_name', str(outpath_heat.stem), merge=False)
        OmegaConf.update(cfg, 'inference.gpu_id', device.index, merge=False)

    if not outpath_heat.is_file():
        _, pred_raster = gdl_inference(cfg.copy())
        torch.cuda.empty_cache()
    else:
        pred_raster_heatmap = _check_rasterio_im_load(str(outpath_heat)).read()
        pred_raster_heatmap = pred_raster_heatmap.squeeze()
        pred_raster = (pred_raster_heatmap > heatmap_threshold)
        create_new_raster_from_base(
            input_raster=str(outpath_heat),
            output_raster=out_raster,
            write_array=pred_raster.astype(int))

    pred_vector_path = gdl_postprocess(cfg.copy())

    logging.info(f"Burning ground truth to raster")
    if num_classes == 1 or (not aoi.attr_field_filter and not aoi.attr_values_filter):
        burn_val = 1
        burn_field = None
    else:
        burn_val = None
        burn_field = aoi.attr_field_filter
    label_raster = vector.mask.footprint_mask(
        df=aoi.label_gdf,
        reference_im=aoi.raster,
        burn_field=burn_field,
        burn_value=burn_val)

    logging.info(f"Computing raster metrics from prediction and ground truth")
    iou = iou_torchmetrics(
        torch.from_numpy(pred_raster),
        torch.from_numpy(label_raster),
        num_classes=num_classes,
        device=device)
    torch.cuda.empty_cache()
    logging.info(f"Result:\nRaster IOU | {iou}")
    metric_per_aoi['iou_raster'] = iou

    logging.info(f"Computing vector metrics from prediction and ground truth")
    metric_vector = iou_per_obj(
        pred=pred_vector_path,
        gt=aoi.label_gdf,
        outfile=out_vector,
        min_iou=min_iou)
    logging.info(f"\nVector metrics: {metric_vector}\nClassified features saved to: {out_vector}")
    metric_vector['iou_threshold'] = min_iou
    metric_per_aoi.update(metric_vector)
    with open(outpath_csv, 'a', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writerows([metric_per_aoi])
    return metric_per_aoi


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
    heatmap_threshold = get_key_def('heatmap_threshold', cfg['inference'], default=50, expected_type=int)
    download_data = get_key_def('download_data', cfg['inference'], default=False, expected_type=bool)

    min_iou = get_key_def('iou_threshold', cfg['evaluate'], default=0.5, expected_type=float)
    max_used_ram = get_key_def('max_used_ram', cfg['training'], default=15)
    max_used_perc = get_key_def('max_used_perc', cfg['training'], default=25)
    gpu_id = get_device_ids(number_requested=1, max_used_ram_perc=max_used_ram, max_used_perc=max_used_perc)
    parallel = get_key_def('parallel', cfg['evaluate'], default=False, expected_type=bool)
    default_gpu_ids = list(get_device_ids(torch.cuda.device_count()).keys())
    parallel_gpu_ids = get_key_def('parallel_gpu_ids', cfg['evaluate'], default=default_gpu_ids)
    if parallel and len(parallel_gpu_ids) == 0:
        raise RuntimeError(f"At least one gpu must be available to use parallel mode.")

    # Save to directory named after model
    root = root / Path(checkpoint).stem
    root.mkdir(exist_ok=True)
    with open_dict(cfg):
        OmegaConf.update(cfg, 'inference.root_dir', str(root), merge=False)
        OmegaConf.update(cfg, 'postprocess.regularization', False, merge=False)  # TODO: softcode
        OmegaConf.update(cfg, 'postprocess.generalization', False, merge=False)  # TODO: softcode

    # Create yaml to use pytorch lightning model management
    if is_url(checkpoint):
        load_state_dict_from_url(url=checkpoint, map_location='cpu', model_dir=models_dir)
        checkpoint = models_dir / Path(checkpoint).name
    checkpoint = gdl2pl_checkpoint(in_pth_path=checkpoint, out_dir=models_dir)
    checkpoint_dict = read_checkpoint(checkpoint, out_dir=models_dir)
    cfg_overridden = override_model_params_from_checkpoint(params=cfg.copy(), checkpoint_params=checkpoint_dict['params'])

    # Dataset params
    bands_requested = get_key_def('bands', cfg_overridden['dataset'], default=("red", "green", "blue"), expected_type=Sequence)
    num_classes = 1  # FIXME override from model: len(cfg.dataset.classes_dict.keys())

    # read input data from csv
    benchmark_raw_data = aois_from_csv(
        csv_path=csv_file,
        bands_requested=bands_requested,
        attr_field_filter=attribute_field,
        attr_values_filter=attr_vals,
        download_data=download_data,
        data_dir=data_dir,
    )

    keys = ["aoi_id", "state_dict", "heatmap_threshold", "iou_raster", "class_id", "iou_field", "TruePos", "FalsePos", "FalseNeg",
            "Precision", "Recall", "F1Score", "iou_threshold"]
    outpath_csv = root / "benchmark.csv"
    if not outpath_csv.is_file():
        with open(outpath_csv, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()

    input_args = []
    for index, aoi in enumerate(benchmark_raw_data):
        if parallel:
            device = torch.device(f"cuda:{parallel_gpu_ids[index % len(parallel_gpu_ids)]}")
            # TODO Remove when solved: https://github.com/dymaxionlabs/dask-rasterio/issues/3
            aoi_raster = aoi.write_multiband_from_singleband_rasters_as_vrt()
            aoi.raster = str(aoi_raster)
            logging.debug(f"Aoi: {aoi.aoi_id} | Device: {device}")
            input_args.append([benchmark_per_aoi, cfg, checkpoint, root, aoi, heatmap_threshold, device, num_classes, min_iou, outpath_csv, keys])
        else:
            device = set_device(gpu_devices_dict=gpu_id)
            benchmark_per_aoi(cfg, checkpoint, root, aoi, heatmap_threshold, device, num_classes, min_iou, outpath_csv, keys)

    if parallel:
        logging.info(f"Will parallelize inferences on {len(parallel_gpu_ids)} gpu(s). Let's heat'em up!")
        with multiprocessing.get_context('spawn').Pool(processes=len(parallel_gpu_ids)) as pool:
            metrics = pool.map_async(map_wrapper, input_args).get()

    logging.info(f"Benchmark report will be saved: {outpath_csv}")


if __name__ == '__main__':
    # test benchmarking with spacenet data

    # test raster iou calculation with test data
    pred_raster_path = "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03_pred.tif"
    # pred_vector_path = "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03_pred.gpkg"
    pred_raster = _check_rasterio_im_load(pred_raster_path).read()

    label_raster_path = "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03_gt.tif"
    label_raster = _check_rasterio_im_load(pred_raster_path).read()

    iou = iou_torchmetrics(torch.from_numpy(pred_raster), torch.from_numpy(label_raster), device='cuda:0')
    print(iou)

    # TODO: which csv to use?
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
