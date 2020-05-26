from sklearn.metrics import classification_report
import numpy as np


def create_metrics_dict(num_classes):
    metrics_dict = {'precision': AverageMeter(), 'recall': AverageMeter(), 'fscore': AverageMeter(),
                    'loss': AverageMeter(), 'iou': AverageMeter()}

    for i in range(0, num_classes):
        metrics_dict['precision_' + str(i)] = AverageMeter()
        metrics_dict['recall_' + str(i)] = AverageMeter()
        metrics_dict['fscore_' + str(i)] = AverageMeter()
        metrics_dict['iou_' + str(i)] = AverageMeter()

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
    min_val = 1e-6
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
    mean_IOU = np.nanmean(ious)
    metric_dict['iou'].update(mean_IOU, batch_size)
    return metric_dict