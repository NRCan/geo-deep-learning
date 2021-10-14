import logging
import os
from omegaconf import OmegaConf
from mlflow import log_metric, exceptions

logger = logging.getLogger(__name__)


def tsv_line(*args):
    return '\t'.join(map(str, args)) + '\n'


class InformationLogger(object):
    def __init__(self, mode):
        self.mode = mode

    def add_values(self, info, epoch, ignore: list = None):
        """Add new information to the logs."""
        ignore = [] if ignore is None else ignore

        for composite_name, value in info.items():
            tokens = composite_name.split('_')
            if len(tokens) == 1:
                # Ordinary metric (non-classwise); e.g. loss, iou, precision
                name = composite_name
            else:
                # Classwise metric; e.g. precision_0, recall_1
                name, _ = tokens

            if name in ignore:
                continue
            else:
                try:
                    log_metric(key=f"{self.mode}_{composite_name}", value=value.avg, step=epoch)
                except exceptions.MlflowException:
                    logging.error(f'Unable to log {composite_name} with the value {value.avg}')


def save_logs_to_bucket(bucket, bucket_output_path, output_path, now, batch_metrics=None):
    if batch_metrics is not None:
        list_log_file = ['metric_val_fscore_averaged', 'metric_val_fscore', 'metric_val_iou',
                         'metric_val_precision_averaged', 'metric_val_precision', 'metric_val_recall_averaged',
                         'metric_val_recall']
    else:
        list_log_file = ['metric_trn_loss', 'metric_val_loss']
    for i in list_log_file:
        if bucket_output_path:
            log_file = os.path.join(output_path, f"{i}.log")
            bucket.upload_file(log_file, os.path.join(bucket_output_path, f"Logs/{now}_{i}.log"))
        else:
            log_file = os.path.join(output_path, f"{i}.log")
            bucket.upload_file(log_file, f"Logs/{now}_{i}.log")
    bucket.upload_file("output.txt", os.path.join(bucket_output_path, f"Logs/{now}_output.txt"))


def dict2path(my_dict, path=None):
    if path is None:
        path = []
    for k, v in my_dict.items():
        newpath = path + [k]
        if isinstance(v, dict):
            for u in dict2path(v, newpath):
                yield u
        else:
            yield newpath, v


def dict_path(param_dict, param_name):
    d2p = OmegaConf.to_container(param_dict[param_name], resolve=True)
    return {
        param_name + '.' + '.'.join(path): v for path, v in dict2path(d2p)
    }
