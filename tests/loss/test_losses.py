from pathlib import Path

from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

from utils.loss import define_loss, verify_weights


class Test_Losses(object):
    """Tests all geo-deep-learning's losses instantiation"""
    def test_set_hyperparameters(self) -> None:
        with initialize(config_path="../../config", job_name="test_ci"):
            for dataset_type in ['binary', 'multiclass']:
                num_classes = 1 if dataset_type == 'binary' else 5
                class_weights = [1/num_classes for i in range(num_classes)]
                verify_weights(num_classes, class_weights)
                for loss_config in Path(to_absolute_path(f"../../config/loss/{dataset_type}")).glob('*.yaml'):
                    cfg = compose(config_name="gdl_config_template",
                                  overrides=[f"loss={dataset_type}/{loss_config.stem}"],
                                  return_hydra_config=True)
                    hconf = HydraConfig()
                    hconf.set_config(cfg)
                    del cfg.loss.is_binary  # prevent exception at instantiation
                    define_loss(loss_params=cfg.loss, class_weights=class_weights)
