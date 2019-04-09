import os
import warnings


def tsv_line(*args):
    return '\t'.join(map(str, args)) + '\n'


class InformationLogger(object):
    def __init__(self, log_folder, mode):
        # List of metrics names
        self.metrics = ['loss']
        self.metrics_classwise = []
        if mode == 'val':
            self.metrics += ['iou']
            self.metrics_classwise += ['precision', 'recall', 'fscore']

        # Dicts of logs
        def open_log(metric_name, fmt_str="metric_{}_{}.log"):
            filename = fmt_str.format(mode, metric_name)
            return open(os.path.join(log_folder, filename), "a", buffering=1)
        self.metric_values = {m: open_log(m) for m in self.metrics}
        self.class_scores = {m: open_log(m) for m in self.metrics_classwise}
        self.averaged_scores = {m: open_log(m, fmt_str="metric_{}_{}_averaged.log") for m in self.metrics_classwise}

    def add_values(self, info, epoch, ignore: list = None):
        """Add new information to the logs."""

        ignore = [] if ignore is None else ignore

        for composite_name, value in info.items():
            tokens = composite_name.split('_')
            if len(tokens) == 1:
                # Ordinary metric (non-classwise); e.g. loss, iou, precision
                name = composite_name
                if name in ignore:
                    continue
                elif name in self.metrics:
                    self.metric_values[name].write(tsv_line(epoch, value.avg))
                elif name in self.metrics_classwise:  # Metrics averaged over classes
                    self.averaged_scores[name].write(tsv_line(epoch, value.avg))
                else:
                    warnings.warn(f'Unknown metric {name}')
            elif len(tokens) == 2:
                # Classwise metric; e.g. precision_0, recall_1
                name, class_idx = tokens
                if name in ignore:
                    continue
                elif name in self.metrics_classwise:
                    self.class_scores[name].write(tsv_line(epoch, class_idx, value.avg))
                else:
                    warnings.warn(f'Unknown metric {name}')


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
