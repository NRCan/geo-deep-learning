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

def accuracy(preds, label):
    """Computes and return the accuracy for a prediction image and a reference image"""
    valid = label.byte()
    # acc_sum = (valid *(preds == label)).sum()
    torch.eq(label, preds, out=valid)
    total_eq = valid.sum()
    total = label.shape[0]
    # acc = float(acc_sum) / (total + 1e-10)
    acc = (float(total_eq) / float(total)) * 100
    print(acc)
    return acc
    

