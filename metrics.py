from sklearn.metrics import classification_report
import torch

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


def Accuracy(preds, label):
    # DEPRECATED.
    """Computes and return the accuracy for a prediction image and a reference image"""
    valid = label.byte()
    torch.eq(label, preds, out=valid)
    total_eq = valid.sum()
    total = label.shape[0]
    acc = (float(total_eq) / float(total)) * 100
    return acc

def ClassificationReport(pred, label, nbClasses):
    """
    Computes precision, recall and f-score for each classes and averaged for all the classes.
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    class_report = classification_report(label, pred)
    content = class_report.split('\n')
    titre = list(filter(None, content[0].split(' ')))
    avgTot = list(filter(None, content[-2].split(' ')))
    prfAvg = [avgTot[3],avgTot[4], avgTot[5]]
    prfScore = []
    for y in range(0,nbClasses):
        prfScore.append(list(filter(None, content[y+2].split(' '))))

    return{'titre': titre, 'prfScore': prfScore, 'prfAvg': prfAvg}
