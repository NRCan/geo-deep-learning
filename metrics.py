import sklearn
from sklearn.metrics import classification_report, precision_recall_fscore_support, jaccard_similarity_score
import torch
import numpy as np

def CreateMetricsdict(num_classes):
    metrics_dict = {'precision': AverageMeter(), 'recall': AverageMeter(), 'fscore': AverageMeter(), 'loss': AverageMeter(), 'iou': AverageMeter()}

    for i in range(0, num_classes):
        metrics_dict['precision_' + str(i)] = AverageMeter()
        metrics_dict['recall_' + str(i)] = AverageMeter()
        metrics_dict['fscore_' + str(i)] = AverageMeter()

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

def ClassificationReport(pred, label, nbClasses, batch_size, metrics_dict):
    """
    Computes precision, recall and f-score for each classes and averaged for all the classes.
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """

    if sklearn.__version__ == '0.19.1':
        # Before 0.20.0, the classification report is a string, therefore not fun to use.

        p, r, f1, s = precision_recall_fscore_support(label, pred, labels=None, average='weighted', sample_weight=None)

        metrics_dict['precision'].update(p, batch_size)
        metrics_dict['recall'].update(r, batch_size)
        metrics_dict['fscore'].update(f1, batch_size)

    elif sklearn.__version__ == '0.20.dev0':
        class_report = classification_report(label, pred, output_dict=True)

        for key, value in class_report.items():
            if key != 'avg / total':

                metrics_dict['precision_' + key].update(value['precision'], batch_size)
                metrics_dict['recall_' + key].update(value['recall'], batch_size)
                metrics_dict['fscore_' + key].update(value['f1-score'], batch_size)

        metrics_dict['precision'].update(class_report['avg / total']['precision'], batch_size)
        metrics_dict['recall'].update(class_report['avg / total']['recall'], batch_size)
        metrics_dict['fscore'].update(class_report['avg / total']['f1-score'], batch_size)

    return metrics_dict

def iou(pred, target, batch_size, metrics_dict):
    # Function to calculate the intersection over union (or jaccard index) between two datasets.
    # The jaccard distance (or dissimilarity) would be 1-iou.

    iou = jaccard_similarity_score(target, pred, normalize=True)
    metrics_dict['iou'].update(iou, batch_size)
    return metrics_dict
