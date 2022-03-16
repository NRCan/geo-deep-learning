from pathlib import Path

import torchvision.models
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

from models.model_choice import set_hyperparameters


class Test_Losses(object):
    """Tests geo-deep-learning's pipeline"""
    def test_set_hyperparameters(self) -> None:
        with initialize(config_path="../../config", job_name="test_ci"):
            for dataset_type in ['binary', 'multiclass']:
                for loss_config in Path(to_absolute_path(f"../../config/loss/{dataset_type}")).glob('*.yaml'):
                    cfg = compose(config_name="gdl_config_template",
                                  overrides=[f"loss={dataset_type}/{loss_config.stem}"],
                                  return_hydra_config=True)
                    hconf = HydraConfig()
                    hconf.set_config(cfg)
                    del cfg.loss.is_binary  # prevent exception at instantiation
                    set_hyperparameters(
                        params={'training': None, 'optimizer': {'params': None},
                                    'scheduler': {'params': None}},
                        model=torchvision.models.resnet18(),
                        checkpoint=None,
                        loss_fn=cfg.loss,
                        optimizer='sgd',
                    )
