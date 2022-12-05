from typing import Dict, Any

import pytorch_lightning as pl
import torch

from dataset.create_dataset import logging
from utils.utils import minmax_scale


class SegmentationDatamodule(pl.LightningDataModule):
    """
    LightningDataModule implementation for the geo-deep-learning datasets
    TODO: use with pytorch lightning. Currently only used for preprocessing.
    """

    def __init__(self, dontcare2backgr: bool = False, dontcare_val: int = None):
        """
        @param dontcare2backgr: if True, dontcare value in label will be replaced by background value (i.e. 0)
        @param dontcare_val: if dontcare2back is True, this value will be replaced by 0 in label.
        """
        super().__init__()
        self.dontcare2backgr = dontcare2backgr
        self.dontcare_val = dontcare_val

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesses a single sample.
        Adapted from torchgeo datamodules
        E.g. https://github.com/microsoft/torchgeo/blob/6f9589dc077839847a2911fb5813eaf908200e17/torchgeo/datamodules/chesapeake.py#L162
        Args:
            sample: sample dictionary containing image and mask
        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        if int(sample["image"].max()) > 255:
            logging.error(f"Image should be 8 bit unsigned (max value of 255). "
                          f"Got max value: {int(sample['image'].max())}.")
        sample["image"] = minmax_scale(sample["image"], orig_range=(0, 255), scale_range=(0, 1))

        if 'mask' in sample and sample['mask'] is not None:  # This can also be used in inference.
            sample['mask'] = sample['mask'].to(torch.long)
            if self.dontcare2backgr:
                sample['mask'][sample['mask'] == self.dontcare_val] = 0
        return sample
