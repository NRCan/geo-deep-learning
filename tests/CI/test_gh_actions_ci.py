from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig

from GDL import run_gdl


class TestTrain(object):
    """Tests geo-deep-learning train script"""
    def test_ci(self) -> None:
        with initialize(config_path="../../config", job_name="test_train_segmentation"):
            cfg = compose(config_name="gdl_config_template")
            for mode in cfg.mode:
                cfg = compose(config_name="gdl_config_template",
                              overrides=[f"mode={mode}", "hydra.job.num=0"], return_hydra_config=True)
                hconf = HydraConfig()
                hconf.set_config(cfg)
                run_gdl(cfg)
