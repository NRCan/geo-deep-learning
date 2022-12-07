import numpy as np
import torch
import torchmetrics.classification
from sklearn.metrics import classification_report
from torch import IntTensor
from torchmetrics import JaccardIndex
import torch.nn.functional as F

min_val = 1e-6


def create_metrics_dict(num_classes, ignore_index=None):

    num_classes = num_classes + 1 if num_classes == 1 else num_classes

    metrics_dict = {'precision': AverageMeter(), 'recall': AverageMeter(), 'fscore': AverageMeter(),
                    'loss': AverageMeter(), 'iou': AverageMeter()}

    for i in range(0, num_classes):
        if ignore_index != i:
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


class AverageMeterFix(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.total_metrics_sum = 0
        self.total_batches = 0
        self.avg = self.average()

    def update(self, val, weight=1):
        self.total_metrics_sum += val
        self.total_batches += 1

    def average(self):
        if self.total_batches != 0:
            return self.total_metrics_sum / self.total_batches
        return 0

    def reset(self):
        self.total_metrics_sum = 0
        self.total_batches = 0


def report_classification(pred, label, batch_size, metrics_dict, ignore_index=-1):
    """Computes precision, recall and f-score for each class and average of all classes.
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    pred = pred.cpu()
    label = label.cpu()
    pred[label == ignore_index] = ignore_index

    # Required to remove ignore_index from scikit-learn's classification report 
    n = max(IntTensor.item(pred.amax()), IntTensor.item(label.amax()))
    labels = np.arange(n+1)

    class_report = classification_report(label, pred, labels=labels, output_dict=True, zero_division=1)

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


def metric_match_device(metric: torchmetrics.Metric, pred: torch.Tensor, label: torch.Tensor):
    """Pushes the metric on devices where predictions and labels already are"""
    if metric.device != pred.device or metric.device != label.device:
        metric.to(pred.device)


def iou(pred, label, batch_size, num_classes, metric_dict, ignore_index=None):
    """Calculate the intersection over union class-wise and mean-iou"""

    num_classes = num_classes + 1 if num_classes == 1 else num_classes
    # Torchmetrics cannot handle ignore_index that are not in range 0 -> num_classes-1.
    # if invalid ignore_index is provided, invalid values (e.g. -1) will be set to 0 
    # and no ignore_index will be used.
    if ignore_index and ignore_index not in range(0, num_classes-1):
        pred[label == ignore_index] = 0
        label[label == ignore_index] = 0
        ignore_index = None
    
    cls_lst = [j for j in range(0, num_classes)]
    if ignore_index is not None:
        cls_lst.remove(ignore_index)

    jaccard = JaccardIndex(num_classes=num_classes, 
                           average='none',
                           ignore_index=ignore_index,
                           absent_score=1)
    metric_match_device(jaccard, pred, label)
    cls_ious = jaccard(pred, label)

    if len(cls_ious) > 1:
        for i in range(len(cls_lst)):
            metric_dict['iou_' + str(cls_lst[i])].update(cls_ious[i], batch_size)
        
    elif len(cls_ious) == 1:
        if f"iou_{cls_lst[0]}" in metric_dict.keys():
            metric_dict['iou_' + str(cls_lst[0])].update(cls_ious, batch_size)

    jaccard_nobg = JaccardIndex(num_classes=num_classes, 
                                average='macro', 
                                ignore_index=0, 
                                absent_score=1)
    metric_match_device(jaccard_nobg, pred, label)
    iou_nobg = jaccard_nobg(pred, label)
    metric_dict['iou_nonbg'].update(iou_nobg.item(), batch_size)

    jaccard = JaccardIndex(num_classes=num_classes, 
                           average='macro', 
                           ignore_index=ignore_index,
                           absent_score=1)
    metric_match_device(jaccard, pred, label)
    mean_iou = jaccard(pred, label)

    metric_dict['iou'].update(mean_iou, batch_size)
    return metric_dict


#### Benchmark Metrics ####
""" Segmentation Metrics from : https://github.com/jeremiahws/dlae/blob/master/metnet_seg_experiment_evaluator.py """


class ComputePixelMetrics():
    '''
    Compute pixel-based metrics between two segmentation masks.
    :param label: (numpy array) reference segmentation mask
    :param pred: (numpy array) predicted segmentation mask
    '''
    __slots__ = 'label', 'pred', 'num_classes'

    def __init__(self, label, pred, num_classes):
        self.label = label
        self.pred = pred
        self.num_classes = num_classes + 1 if num_classes == 1 else num_classes

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


def calculate_batch_iou(predictions, gts, n_classes, metric_dict):
    batch_size = predictions.shape[0]

    def _fast_hist(label_pred, label_true, n_classes):
        n_classes = n_classes + 1 if n_classes == 1 else n_classes

        mask = (label_true >= 0) & (label_true < n_classes)
        hist = np.bincount(
            n_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_classes ** 2).reshape(n_classes, n_classes)
        return hist

    batch_hist = np.zeros((n_classes, n_classes)) if n_classes > 1 else np.zeros((n_classes+1, n_classes+1))
    label_true = gts.clone().cpu().detach().numpy().copy()

    if n_classes == 1:
        label_pred = torch.sigmoid(predictions).clone().cpu().detach().numpy()
        ones_array = np.ones(shape=label_pred.shape, dtype=np.uint8)
        bg_pred = ones_array - label_pred
        label_pred = np.concatenate([bg_pred, label_pred], axis=1)
        cls_list = [str(cls) for cls in range(2)]
    elif n_classes > 1:
        label_pred = F.softmax(predictions.clone().cpu().detach().numpy(), dim=1)
        cls_list = [str(cls) for cls in range(n_classes)]
    else:
        raise ValueError(f'Number of classes is less than 1 == {n_classes}')
    label_pred = np.array(np.argmax(label_pred, axis=1)).astype(np.uint8)

    for lp, lt in zip(label_pred, label_true):
        batch_hist += _fast_hist(lp.flatten(), lt.flatten(), n_classes)
    # acc = np.diag(batch_hist).sum() / batch_hist.sum()
    iu = np.diag(batch_hist) / (batch_hist.sum(axis=1) + batch_hist.sum(axis=0) - np.diag(batch_hist))
    mean_iu = np.nanmean(iu)
    mean_iu_nobg = np.nanmean(iu[1:])

    metric_dict['iou'].update(mean_iu, batch_size)
    metric_dict['iou_nonbg'].update(mean_iu_nobg, batch_size)

    for i, cls_lbl in enumerate(cls_list):
        metric_dict['iou_' + cls_lbl].update(iu[i], batch_size)

    return metric_dict

