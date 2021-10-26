import logging
import os
from pathlib import Path
from typing import Union

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
                    logging.exception(f'Unable to log {value.avg}')


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


def set_logging(console_level: str = 'WARNING', logfiles_dir: Union[str, Path] = None, logfiles_prefix: str = 'log',
                conf_path: Union[str, Path] = 'utils/logging.conf'):
    """
    Configures logging with provided ".conf" file, console level, output paths.
    @param conf_path: Path to ".conf" file with loggers, handlers, formatters, etc.
    @param console_level: Level of logging to output to console. Defaults to "WARNING"
    @param logfiles_dir: path where output logs will be written
    @return:
    """
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    if not logfiles_dir:
        logging.basicConfig(format='%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=logging.INFO)
        return
    conf_path = Path(conf_path).absolute()
    if not conf_path.is_file():
        raise FileNotFoundError(f'Invalid logging configuration file')
    log_config_path = Path(conf_path).absolute()
    out = Path(logfiles_dir) / logfiles_prefix
    logging.config.fileConfig(log_config_path, defaults={'logfilename': f'{out}.log',
                                                         'logfilename_error': f'{out}_error.log',
                                                         'logfilename_debug': f'{out}_debug.log',
                                                         'console_level': console_level})