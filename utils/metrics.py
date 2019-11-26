from sklearn.metrics import classification_report, jaccard_similarity_score


def create_metrics_dict(num_classes):
    metrics_dict = {'precision': AverageMeter(), 'recall': AverageMeter(), 'fscore': AverageMeter(),
                    'loss': AverageMeter(), 'iou': AverageMeter()}

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


def report_classification(pred, label, batch_size, metrics_dict, ignore_index=-1):
    """Computes precision, recall and f-score for each class and average of all classes.
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    class_report = classification_report(label.cpu(), pred.cpu(), output_dict=True)

    class_score = {}
    for key, value in class_report.items():
        if key not in ['micro avg', 'macro avg', 'weighted avg', 'accuracy'] and not str(ignore_index): #TODO test this with ignore index. Make sure it works.
            class_score[key] = value

            metrics_dict['precision_' + key].update(class_score[key]['precision'], batch_size)
            metrics_dict['recall_' + key].update(class_score[key]['recall'], batch_size)
            metrics_dict['fscore_' + key].update(class_score[key]['f1-score'], batch_size)

    metrics_dict['precision'].update(class_report['weighted avg']['precision'], batch_size)
    metrics_dict['recall'].update(class_report['weighted avg']['recall'], batch_size)
    metrics_dict['fscore'].update(class_report['weighted avg']['f1-score'], batch_size)

    return metrics_dict


def iou(pred, target, batch_size, metrics_dict):
    """Calculate the intersection over union (or Jaccard index) between two datasets.
    The Jaccard distance (or dissimilarity) would be 1-iou.
    """
    iou = jaccard_similarity_score(target.cpu(), pred.cpu(), normalize=True)
    metrics_dict['iou'].update(iou, batch_size)
    return metrics_dict
