# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""
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
    # valid = (label >= 0)
    # print(preds.shape)
    # print(label.shape)
    # flt_label = label.float()
    acc_sum = (preds == label).sum()
    total_sum = label.shape
    acc = float(acc_sum) / (total_sum[0]) * 100
    return acc
    

