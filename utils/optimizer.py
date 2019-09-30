import torch.optim as optim
from .adabound import AdaBound, AdaBoundW


def create_optimizer(params, mode='adam', base_lr=1e-3, weight_decay=4e-5):
    if mode == 'adam':
        optimizer = optim.Adam(params, lr=base_lr)
    elif mode == 'sgd':
        optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    elif mode == 'adabound':
        optimizer = AdaBound(params, lr=base_lr, final_lr=base_lr*100, gamma=1e-3, eps=1e-8, weight_decay=weight_decay)
    elif mode == 'adaboundw':
        optimizer = AdaBoundW(params, lr=base_lr, final_lr=base_lr * 100, gamma=1e-3, eps=1e-8, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"The optimizer should be one of: adam, sgd or adabound. The provided value is {mode}")

    return optimizer
