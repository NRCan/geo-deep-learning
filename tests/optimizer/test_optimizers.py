from pathlib import Path

import torchvision.models
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path, instantiate

from utils.loss import define_loss, verify_weights


class TestOptimizersZoo(object):
    """Tests all geo-deep-learning's optimizers instantiation"""
    def test_instantiate_optimizer(self) -> None:
        with initialize(config_path="../../config", job_name="test_ci"):
            for optimizer_config in Path(to_absolute_path(f"../../config/optimizer")).glob('*.yaml'):
                cfg = compose(config_name="gdl_config_template",
                              overrides=[f"optimizer={optimizer_config.stem}"],
                              return_hydra_config=True)
                hconf = HydraConfig()
                hconf.set_config(cfg)
                model = torchvision.models.resnet18()
                instantiate(cfg.optimizer, params=model.parameters())
