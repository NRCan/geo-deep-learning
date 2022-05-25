import logging
from pathlib import Path
from typing import Sequence

from pandas.io.common import is_url
from torch.hub import load_state_dict_from_url

from dataset.aoi import aois_from_csv
from utils.metrics import iou_per_obj
from utils.utils import get_key_def, gdl2pl_checkpoint, read_checkpoint, \
    override_model_params_from_checkpoint
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
            
    # Create yaml to use pytorch lightning model management
    if is_url(checkpoint):
        load_state_dict_from_url(url=checkpoint, map_location='cpu', model_dir=models_dir)
        checkpoint = models_dir / Path(checkpoint).name
    checkpoint = gdl2pl_checkpoint(in_pth_path=checkpoint, out_dir=models_dir)
    checkpoint_dict = read_checkpoint(checkpoint, out_dir=models_dir)
    cfg_overridden = override_model_params_from_checkpoint(params=cfg.copy(), checkpoint_params=checkpoint_dict['params'])

    # Dataset params
    bands_requested = get_key_def('bands', cfg_overridden['dataset'], default=("red", "blue", "green"), expected_type=Sequence)

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
        logging.info(f"Benchmarking: {aoi.aoi_id}")
        # inference on each raster
        cfg['inference']['input_stac_item'] = aoi.raster_raw_input
        cfg['inference']['output_name'] = aoi.aoi_id + '_BUIL'
        # pred_raster, pred_raster_path = gdl_inference(cfg)
        # compute raster metrics from prediction and ground truth
        # TODO: where to reference numpy arrays?
        #jaccard = JaccardIndex(num_classes=num_classes, ignore_index=None, threshold=0.5, multilabel=True)

        # compute vector metrics from prediction and ground truth
        pred_vector_path = gdl_postprocess(cfg)

        metric = iou_per_obj(
            pred=pred_vector_path,
            gt=aoi.label,
            attr_field=aoi.attr_field_filter,
            attr_vals=[cfg['dataset']['classes_dict']['BUIL']],
            aoi_id=aoi.aoi_id,
            aoi_categ=None,  # TODO: add category for human-readable report
            gt_clip_bounds=None)
        metrics.append(metric)
        print(metrics)
        break



