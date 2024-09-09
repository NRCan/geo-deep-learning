from pathlib import Path

from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from torchgeo.datasets.utils import extract_archive

from GDL import run_gdl


class Test_GH_Actions(object):
    """Tests geo-deep-learning's pipeline"""
    def test_ci(self) -> None:
        data_dir = "data"
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        extract_archive(from_path="tests/data/spacenet.zip")
        extract_archive(from_path="tests/data/new_brunswick_aerial.zip")
        extract_archive(from_path="tests/data/massachusetts_buildings_kaggle.zip")

        with initialize(config_path="../../config", job_name="test_ci"):
            cfg = compose(config_name="gdl_config_template")
            modes = cfg.mode
            for dataset_type in ['binary', 'multiclass']:
                for mode in modes:
                    cfg = compose(config_name="gdl_config_template",
                                  overrides=[f"mode={mode}",
                                             f"dataset=test_ci_segmentation_{dataset_type}",
                                             f"inference=default_{dataset_type}",
                                             f"training.num_gpus=0",
                                             f"loss={dataset_type}/lovasz",
                                             "inference.output_path=./tst.tif",
                                             "hydra.job.num=0"],
                                  return_hydra_config=True)
                    hconf = HydraConfig()
                    hconf.set_config(cfg)
                    run_gdl(cfg)
