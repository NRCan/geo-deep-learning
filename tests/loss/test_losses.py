from pathlib import Path

import torch
import os
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

from utils.loss import define_loss, verify_weights


class TestLossesZoo(object):
    """Tests all geo-deep-learning's losses instantiation"""
    def test_verify_weights(self) -> None:
        with initialize(config_path="../../config", job_name="test_ci"):
            for dataset_type in ['binary', 'multiclass']:
                num_classes = 1 if dataset_type == 'binary' else 5
                class_weights = [1/num_classes for i in range(num_classes)]
                verify_weights(num_classes, class_weights)
                
    def test_define_loss(self) -> None:
        with initialize(config_path="../../config", job_name="test_ci"):
            for dataset_type in ['binary', 'multiclass']:
                num_classes = 1 if dataset_type == 'binary' else 5
                class_weights = [1/num_classes for i in range(num_classes)]
                for loss_config in Path(to_absolute_path(f"config/loss/{dataset_type}")).glob('*.yaml'):
                    cfg = compose(config_name="gdl_config_template",
                                  overrides=[f"loss={dataset_type}/{loss_config.stem}"],
                                  return_hydra_config=True)
                    hconf = HydraConfig()
                    hconf.set_config(cfg)
                    del cfg.loss.is_binary  # prevent exception at instantiation
                    define_loss(loss_params=cfg.loss, class_weights=class_weights)
    
    def test_binary_multi_loss(self) -> None:
        with initialize(config_path="../../config", job_name="test_ci"):
            for dataset_type in ['binary', 'multiclass']:
                num_classes = 1 if dataset_type == 'binary' else 5
                class_weights = [1/num_classes for i in range(num_classes)]
                for loss_config in Path(to_absolute_path(f"config/loss/{dataset_type}")).glob('*.yaml'):
                    cfg = compose(config_name="gdl_config_template",
                                  overrides=[f"loss={dataset_type}/{loss_config.stem}"],
                                  return_hydra_config=True)
                    hconf = HydraConfig()
                    hconf.set_config(cfg)
                    del cfg.loss.is_binary  # prevent exception at instantiation
                    criterion = define_loss(
                        loss_params=cfg.loss, class_weights=class_weights
                    )
                    # test if binary and multiclass work
                    outputs = torch.ones(1, num_classes, 256, 256)
                    labels = torch.ones(1, 256, 256, dtype=torch.long)
                    loss = criterion(outputs, labels) if num_classes > 1 else criterion(outputs, labels.unsqueeze(1).float())
