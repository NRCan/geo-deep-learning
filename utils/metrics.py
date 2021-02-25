import numpy as np
from sklearn.metrics import classification_report, matthews_corrcoef, recall_score
from math import sqrt

min_val = 1e-6
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
        # minimum value added to avoid Zero division
        iou = (intersection + min_val) / (union + min_val)
        ious.append(iou)
        metric_dict['iou_' + str(i)].update(iou.item(), batch_size)
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
        classes= []
        for i in range(self.num_classes):
            c_label = self.label.ravel() == i
            if c_label.sum() == 0:
                classes.append(np.nan)
                continue
            c_pred = self.pred.ravel()== i
            m = metric_func(c_label, c_pred)
            metric[metric_func.__name__ + '_' + str(i)] = m
            classes.append(m)
        mean_m = np.nanmean(classes)
        metric['macro_avg_'+ metric_func.__name__] = mean_m
        micro = metric_func(self.label.ravel(), self.pred.ravel())
        metric['micro_avg_'+ metric_func.__name__] = micro
        return metric

    @staticmethod
    def jaccard(label, pred):
        '''
        :return: IOU
        '''
        both = np.logical_and(label, pred)
        either = np.logical_or(label, pred)
        ji = (np.sum(both) + min_val) / (np.sum(either) + min_val)

        return ji

    @staticmethod
    def dice(label, pred):
        '''
        :return: Dice similarity coefficient
        '''
        both = np.logical_and(label, pred)
        dsc = 2 * int(np.sum(both)) / (int(np.sum(label)) + int(np.sum(pred)))

        return dsc

    @staticmethod
    def precision(label, pred):
        '''
        :return: precision
        '''
        tp = int(np.sum(np.logical_and(label, pred)))
        fp = int(np.sum(np.logical_and(pred, np.logical_not(label))))
        p = (tp + min_val) / (tp + fp + min_val)

        return p

    @staticmethod
    def recall(label, pred):
        '''
        :return: recall
        '''
        tp = int(np.sum(np.logical_and(label, pred)))
        fn = int(np.sum(np.logical_and(label, np.logical_not(pred))))
        r = (tp + min_val) / (tp + fn + min_val)

        return r

    @staticmethod
    def matthews(label, pred):
        '''
        :return: Matthews correlation coefficient
        '''
        tp = int(np.sum(np.logical_and(label, pred)))
        fp = int(np.sum(np.logical_and(pred, np.logical_not(label))))
        tn = int(np.sum(np.logical_and(np.logical_not(label), np.logical_not(pred))))
        fn = int(np.sum(np.logical_and(label, np.logical_not(pred))))
        mcc = (tp * tn - fp * fn) / (sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

        return mcc

    @staticmethod
    def accuracy(label, pred):
        '''
        :return: accuracy
        '''
        acc = np.mean(pred == label)
        return acc
