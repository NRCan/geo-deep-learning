class MetricTracker(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(preds, label):
    """Computes and return the accuracy for a prediction image and a reference image"""
    acc_sum = (preds == label).sum()
    total = label.shape[0]
    acc = float(acc_sum) / (total) * 100
    return acc
    

