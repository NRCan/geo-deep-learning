import sys
import os
import logging
import torch.nn as nn
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import torch
from typing import Union, Optional
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only

class ModelHelpers:
    
    @staticmethod
    def batchnorm2d(bn_type: Union[str, "torch_sync_bn", "torch_bn"] = "torch_bn"):
        if bn_type == "torch_bn":
            return nn.BatchNorm2d
        if bn_type == "torch_sync_bn":
            return nn.SyncBatchNorm
        
    @staticmethod
    def BNReLU(ch: torch.Tensor):
        batchnorm = ModelHelpers.batchnorm2d()
        return nn.Sequential(
            batchnorm(ch),
            nn.ReLU())
        
    @rank_zero_only        
    @staticmethod
    def load_url(url: str, download: bool):
        model_dir = Path.home() / ".cache" / "torch" / "checkpoints"
        if not model_dir.is_dir():
            Path.mkdir(model_dir, parents=True)
        filename = url.split('/')[-1]
        cached_file = model_dir.joinpath(filename)
        if not cached_file.is_file() and download:
            logging.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, str(cached_file))
        return cached_file