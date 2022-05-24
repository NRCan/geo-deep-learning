import logging

import geopandas as gpd

from utils.utils import read_csv, get_key_def
from inference_segmentation import main as gdl_inference
from postprocess_segmentation import main as gdl_postprocess
from solaris.eval.base import Evaluator
from torchmetrics.classification.jaccard import JaccardIndex

csv_file = "/home/remi/Downloads/benchmark_ccmeo_geoai.csv"

def main(cfg):
    logging.debug(f"\nSetting inference parameters")
    # Main params
    #csv_file = get_key_def('raw_data_csv', cfg['dataset'], to_path=True, validate_path_exists=True)
    root = get_key_def('root_dir', cfg['inference'], default="inference", to_path=True)
    root.mkdir(exist_ok=True)
    num_classes = len(cfg.dataset.classes_dict.keys())

    # read input data from csv
    # TODO read stac item (branch 223)
    benchmark_raw_data = read_csv(csv_file)

    for aoi in benchmark_raw_data:
        logging.info(f"Benchmarking: {aoi}")
        # inference on each raster
        cfg['inference']['input_stac_item'] = aoi['tif']
        pred_raster, pred_raster_path = gdl_inference(cfg)
        # compute raster metrics from prediction and ground truth
        # TODO: where to reference numpy arrays?
        #jaccard = JaccardIndex(num_classes=num_classes, ignore_index=None, threshold=0.5, multilabel=True)

        # compute vector metrics from prediction and ground truth
        pred_vector_path = gdl_postprocess(cfg)

        evaluator = Evaluator(pred_vector_path)
        evaluator.eval_iou_return_GDFs(miniou=0.5)
        break



