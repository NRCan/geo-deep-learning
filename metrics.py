import argparse
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Union, List

import numpy as np
import rasterio
from sklearn.metrics import classification_report

import geopandas as gpd
import pandas as pd
from solaris_gdl.eval.base import Evaluator
from tqdm import tqdm

from data_to_tiles import filter_gdf
from utils.readers import read_parameters
from utils.utils import get_git_hash, get_key_def, read_csv


min_val = 1e-6
def create_metrics_dict(num_classes):
    metrics_dict = {'precision': AverageMeter(), 'recall': AverageMeter(), 'fscore': AverageMeter(),
                    'loss': AverageMeter(), 'iou': AverageMeter()}

    for i in range(0, num_classes):
        metrics_dict['precision_' + str(i)] = AverageMeter()
        metrics_dict['recall_' + str(i)] = AverageMeter()
        metrics_dict['fscore_' + str(i)] = AverageMeter()
        metrics_dict['iou_' + str(i)] = AverageMeter()

    # Add overall non-background iou metric
    metrics_dict['iou_nonbg'] = AverageMeter()

    return metrics_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def report_classification(pred, label, batch_size, metrics_dict, ignore_index=-1):
    """Computes precision, recall and f-score for each class and average of all classes.
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    class_report = classification_report(label.cpu(), pred.cpu(), output_dict=True)

    class_score = {}
    for key, value in class_report.items():
        if key not in ['micro avg', 'macro avg', 'weighted avg', 'accuracy'] and key != str(ignore_index):
            class_score[key] = value

            metrics_dict['precision_' + key].update(class_score[key]['precision'], batch_size)
            metrics_dict['recall_' + key].update(class_score[key]['recall'], batch_size)
            metrics_dict['fscore_' + key].update(class_score[key]['f1-score'], batch_size)

    metrics_dict['precision'].update(class_report['weighted avg']['precision'], batch_size)
    metrics_dict['recall'].update(class_report['weighted avg']['recall'], batch_size)
    metrics_dict['fscore'].update(class_report['weighted avg']['f1-score'], batch_size)

    return metrics_dict


def iou(pred, label, batch_size, num_classes, metric_dict, only_present=True):
    """Calculate the intersection over union class-wise and mean-iou"""
    ious = []
    pred = pred.cpu()
    label = label.cpu()
    for i in range(num_classes):
        c_label = label == i
        if only_present and c_label.sum() == 0:
            ious.append(np.nan)
            continue
        c_pred = pred == i
        intersection = (c_pred & c_label).float().sum()
        union = (c_pred | c_label).float().sum()
        iou = (intersection + min_val) / (union + min_val)  # minimum value added to avoid Zero division
        ious.append(iou)
        metric_dict['iou_' + str(i)].update(iou.item(), batch_size)

    # Add overall non-background iou metric
    c_label = (1 <= label) & (label <= num_classes - 1)
    c_pred = (1 <= pred) & (pred <= num_classes - 1)
    intersection = (c_pred & c_label).float().sum()
    union = (c_pred | c_label).float().sum()
    iou = (intersection + min_val) / (union + min_val)  # minimum value added to avoid Zero division
    metric_dict['iou_nonbg'].update(iou.item(), batch_size)

    mean_IOU = np.nanmean(ious)
    if (not only_present) or (not np.isnan(mean_IOU)):
        metric_dict['iou'].update(mean_IOU, batch_size)
    return metric_dict

#### Benchmark Metrics ####
""" Segmentation Metrics from : https://github.com/jeremiahws/dlae/blob/master/metnet_seg_experiment_evaluator.py """

class ComputePixelMetrics():
    '''
    Compute pixel-based metrics between two segmentation masks.
    :param label: (numpy array) reference segmentaton mask
    :param pred: (numpy array) predicted segmentaton mask
    '''
    __slots__ = 'label', 'pred', 'num_classes'

    def __init__(self, label, pred, num_classes):
        self.label = label
        self.pred = pred
        self.num_classes = num_classes

    def update(self, metric_func):
        metric = {}
        classes = []
        for i in range(self.num_classes):
            c_label = self.label == i
            c_pred = self.pred == i
            m = metric_func(c_label, c_pred)
            classes.append(m)
            metric[metric_func.__name__ + '_' + str(i)] = classes[i]
        mean_m = np.nanmean(classes)
        metric['macro_avg_' + metric_func.__name__] = mean_m
        metric['micro_avg_' + metric_func.__name__] = metric_func(self.label, self.pred)
        return metric

    @staticmethod
    def iou(label, pred):
        '''
        :return: IOU
        '''
        intersection = (pred & label).sum()
        union = (pred | label).sum()
        iou = (intersection + min_val) / (union + min_val)

        return iou

    @staticmethod
    def dice(label, pred):
        '''
        :return: Dice similarity coefficient
        '''
        intersection = (pred & label).sum()
        dice = (2 * intersection) / ((label.sum()) + (pred.sum()))

        return dice

def iou_per_obj(pred: Union[str, Path], gt:Union[str, Path], attr_field: str = None, attr_vals: List = None,
                aoi_id: str = None, aoi_categ: str = None, gt_clip_bounds = None):
    """
    Calculate iou per object by comparing vector ground truth and vectorized prediction
    @param pred:
    @param gt:
    @param attr_field:
    @param attr_vals:
    @param aoi_id:
    @return:
    """
    if not aoi_id:
        aoi_id = Path(pred).stem
    # filter out non-buildings
    gt_gdf = gpd.read_file(gt, bbox=gt_clip_bounds)
    gt_gdf_filtered = filter_gdf(gt_gdf, attr_field, attr_vals).copy(deep=True)
    #
    evaluator = Evaluator(ground_truth_vector_file=gt_gdf_filtered)

    evaluator.load_proposal(pred, conf_field_list=None)
    scoring_dict_list, TP_gdf, FN_gdf, FP_gdf = evaluator.eval_iou_return_GDFs(calculate_class_scores=False)

    if TP_gdf is not None:
        TP_gdf.to_file(pred, layer='True_Pos', driver="GPKG")
    if FN_gdf is not None:
        FN_gdf.to_file(pred, layer='False_Neg', driver="GPKG")
    if FP_gdf is not None:
        FP_gdf.to_file(pred, layer='False_Pos', driver="GPKG")

    scoring_dict_list[0] = OrderedDict(scoring_dict_list[0])
    scoring_dict_list[0]['aoi'] = aoi_id
    scoring_dict_list[0].move_to_end('aoi', last=False)
    scoring_dict_list[0]['category'] = aoi_categ
    scoring_dict_list[0].move_to_end('category', last=False)
    logging.info(scoring_dict_list[0])
    return scoring_dict_list[0]


def main(params):
    """
    -------
    :param params: (dict) Parameters found in the yaml config file.
    """
    start_time = time.time()

    # mlflow logging
    mlflow_uri = get_key_def('mlflow_uri', params['global'], default="./mlruns")
    exp_name = get_key_def('mlflow_experiment_name', params['global'], default='gdl-training', expected_type=str)

    # MANDATORY PARAMETERS
    default_csv_file = Path(get_key_def('preprocessing_path', params['global'], ''),
                            exp_name, f"inference_sem_seg_{exp_name}.csv")
    img_dir_or_csv = get_key_def('img_dir_or_csv_file', params['inference'], default_csv_file, expected_type=str)
    state_dict = get_key_def('state_dict_path', params['inference'], expected_type=str)
    num_classes = get_key_def('num_classes', params['global'], expected_type=int)
    num_bands = get_key_def('number_of_bands', params['global'], expected_type=int)
    attr_vals = get_key_def('target_ids', params['sample'], [4], List) if 'sample' in params.keys() else [4]

    # OPTIONAL PARAMETERS
    # basics
    debug = get_key_def('debug_mode', params['global'], False)
    parallel = get_key_def('parallelize', params['inference'], default=True, expected_type=bool)
    dryrun = get_key_def('dryrun', params['inference'], default=False, expected_type=bool)

    # SETTING OUTPUT DIRECTORY
    working_folder = Path(state_dict).parent / f'inference_{num_bands}bands'
    if not working_folder.is_dir():
        raise NotADirectoryError(f"Couldn't find source inference directory: {working_folder}")

    working_folder_pp = working_folder.parent / f'{working_folder.stem}_post-process'
    Path.mkdir(working_folder_pp, exist_ok=True)
    # add git hash from current commit to parameters if available. Parameters will be saved to hdf5s
    params['global']['git_hash'] = get_git_hash()

    # See: https://docs.python.org/2.4/lib/logging-config-fileformat.html
    log_config_path = Path('utils/logging.conf').absolute()
    console_level_logging = 'INFO' if not debug else 'DEBUG'
    log_file_prefix = 'metrics'
    logging.config.fileConfig(log_config_path, defaults={'logfilename': f'{working_folder_pp}/{log_file_prefix}.log',
                                                         'logfilename_error':
                                                             f'{working_folder_pp}/{log_file_prefix}_error.log',
                                                         'logfilename_debug':
                                                             f'{working_folder_pp}/{log_file_prefix}_debug.log',
                                                         'console_level': console_level_logging})

    if Path(img_dir_or_csv).suffix == '.csv':
        inference_srcdata_list = read_csv(Path(img_dir_or_csv))
    elif Path(img_dir_or_csv).is_dir():
        raise ValueError(f'Metrics.py requires csv list of images and ground truths. '
                         f'Got invalid "img_dir_or_csv" parameter: "{img_dir_or_csv}"')

    metrics = []
    for info in tqdm(inference_srcdata_list):
        if info['dataset'] == 'tst':
            aoi_id = Path(info["tif"]).stem
            region = info['aoi'][0].capitalize()
            gt = Path(info['gpkg'])
            gt_clip_bounds = rasterio.open(info["tif"], 'r').bounds
            pred_glob = list(working_folder_pp.glob(f'{aoi_id}_*raw.gpkg'))
            if len(pred_glob) == 1:
                pred = pred_glob[0]
                logging.info(f'\nImage: {info["tif"]}\nGround truth: {gt}\nPrediction from glob: {pred}')
                if not dryrun:
                    metric = iou_per_obj(pred, gt, info['attribute_name'], attr_vals, aoi_id, region, gt_clip_bounds)
                    metrics.append(metric)
            else:
                logging.critical(f"No single vectorized prediction file found to match ground truth {gt}.\n"
                                 f"Got: {pred_glob}")
    df = pd.DataFrame(metrics)
    df_md = df.to_markdown()
    logging.info(f'\n{df_md}')
    out_metrics = working_folder_pp / 'buildings_metrics.csv'
    df.to_csv(out_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Buildings post processing')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.ParamFile)
    print(f'\n\nStarting post-processing with {args.ParamFile}\n\n')
    main(params)