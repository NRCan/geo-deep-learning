from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from torchgeo.datasets.utils import extract_archive

from GDL import run_gdl
from dataset.aoi import AOI
from utils.utils import read_csv
from verify_segmentation import verify_per_aoi


class TestVerify(object):
    def test_verify_per_aoi(self):
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        data = read_csv("tests/sampling/sampling_segmentation_multiclass_ci.csv")
        aoi = AOI(raster=data[0]['tif'], label=data[0]['gpkg'], split=data[0]['split'])
        aoi_dict, error = verify_per_aoi(
            aoi=aoi,
            output_raster_plots=True,
            output_raster_stats=True,
            extended_label_stats=True,
            output_report_dir="dataset",
        )
        assert aoi_dict['raster_area'] == 23870.0
        assert aoi_dict['label_features_filtered_mean_exterior_vertices_nb'] == 36.093023255813954
        assert aoi_dict['band_0_mean'] == 159.36075617930456

    def test_verify_segmentation_parallel(self):
        with initialize(config_path="../config", job_name="test_ci"):
            cfg = compose(config_name="gdl_config_template",
                          overrides=[f"mode=verify",
                                     f"dataset=test_ci_segmentation_multiclass",
                                     f"verify.multiprocessing=True",
                                     "hydra.job.num=0"],
                          return_hydra_config=True)
            hconf = HydraConfig()
            hconf.set_config(cfg)
            run_gdl(cfg)

