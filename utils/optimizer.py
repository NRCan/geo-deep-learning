import torch.optim as optim
# from .scheduler import CosineWithRestarts # TODO: check if scheduler if useful. will have to merge with current implementation
from .adabound import AdaBound


def create_optimizer(params, mode='adam', base_lr=1e-3, weight_decay=0): #, t_max=10):
    if mode == 'adam':
        optimizer = optim.Adam(params, lr=base_lr)
    elif mode == 'sgd':
        optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=4e-5)
    elif mode == 'adabound':
        optimizer = AdaBound(params, lr=base_lr, final_lr=base_lr*100, gamma=1e-3, eps=1e-8, weight_decay=4e-5) #TODO: set correct final_lr
    else:
        raise NotImplementedError(f"The optimizer should be one of: adam, sgd or adabound. The provided value is {mode}")

    # scheduler = CosineWithRestarts(optimizer, t_max)

    return optimizer #, scheduler
