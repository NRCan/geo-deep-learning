import shutil
from pathlib import Path

import numpy as np
import rasterio
from omegaconf import DictConfig
from torchgeo.datasets.utils import extract_archive

from tiling_segmentation import annot_percent, main
from utils.utils import read_csv


class TestTiler(object):
    def test_filter_tile_pair(self):
        pass

    def test_get_burn_gt_tile_pair(self):
        pass

    def test_filter_and_burn_dataset(self):
        pass

    def test_tiler_inference(self):
        # FIXME: KeyError with tiler.datasets if aoi.split == inference.
        pass


class TestTiling(object):
    def test_outputted_chips(self):
        """Tests the tiling process to ensure all expected tiles are created"""
        pass
        # TODO
        # data_dir = "data/tiles"
        # cfg = {
        #     "general": {"project_name": "test"},
        #     "debug": True,
        #     "dataset": {
        #         "bands": ["R", "G", "B"],
        #         "raw_data_dir": data_dir,
        #         "raw_data_csv": "tests/tiling/tiling_segmentation_multiclass_ci.csv",
        #         "attribute_field": "Quatreclasses",
        #         "attribute_values": [4]
        #     },
        #     "tiling": {"tile_size": 32},
        # }
        # cfg = DictConfig(cfg)
        # main(cfg)
        # out_labels = Path("data/tiles/test/tiles32_RGBbands/trn/23322E759967N_clipped_1m_1of2/labels_burned")
        # assert out_labels.exists()

    def test_min_annotated_percent_filter(self):
        """Tests the minimum annotated percent filter"""
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        data = read_csv("tests/tiling/tiling_segmentation_multiclass_ci.csv")
        iterator = iter(data)
        row = next(iterator)
        annot_out = annot_percent(row['tif'], row['gpkg'])
        assert int(annot_out) == 330  # ground truth larger than raster

    def test_attribute_filter_continuous(self):
        """Tests the attribute field and values filter with continuous output"""
        data_dir = "data/tiles"
        cfg = {
            "general": {"project_name": "test"},
            "debug": True,
            "dataset": {
                "bands": ["R", "G", "B"],
                "raw_data_dir": data_dir,
                "raw_data_csv": "tests/tiling/tiling_segmentation_multiclass_ci.csv",
                "attribute_field": "Quatreclasses",
                "attribute_values": [4]
            },
            "tiling": None,
        }
        cfg = DictConfig(cfg)
        main(cfg)
        out_labels = Path("data/tiles/test/tiles512_RGBbands/trn/23322E759967N_clipped_1m_1of2/labels_burned")
        assert out_labels.is_dir()
        label = next(out_labels.iterdir())
        label_np = rasterio.open(label).read()
        assert list(np.unique(label_np)) == [0, 1]

    def test_attribute_filter_discontinuous(self):
        """Tests the attribute field and values filter with discontinuous output pixel values"""
        data_dir = "data/tiles"
        project_name = "test"
        cfg = {
            "general": {"project_name": project_name},
            "debug": True,
            "dataset": {
                "bands": ["R", "G", "B"],
                "raw_data_dir": data_dir,
                "raw_data_csv": "tests/tiling/tiling_segmentation_multiclass_ci.csv",
                "attribute_field": "Quatreclasses",
                "attribute_values": [4]
            },
            "tiling": {
                "continuous_values": False,
            },
        }
        cfg = DictConfig(cfg)
        main(cfg)
        out_labels = Path("data/tiles/test/tiles512_RGBbands/trn/23322E759967N_clipped_1m_1of2/labels_burned")
        assert out_labels.is_dir()
        label = next(out_labels.iterdir())
        label_np = rasterio.open(label).read()
        assert list(np.unique(label_np)) == [0, 4]
        shutil.rmtree(Path(data_dir) / project_name)

    def test_tiling_segmentation_parallel(self):
        data_dir = "data/tiles"
        project_name = "test_parallel"
        cfg = {
            "general": {"project_name": project_name},
            "debug": True,
            "dataset": {
                "bands": ["R", "G", "B"],
                "raw_data_dir": data_dir,
                "raw_data_csv": "tests/tiling/tiling_segmentation_multiclass_ci.csv",
                "attribute_field": "Quatreclasses",
                "attribute_values": [1, 2, 3, 4]
            },
            "tiling": {
                "multiprocessing": True,
                "tile_size": 32,
            }
        }
        cfg = DictConfig(cfg)
        main(cfg)
        out_labels = [
            Path("data/tiles/test_parallel/tiles32_RGBbands/trn/23322E759967N_clipped_1m_1of2/labels_burned"),
            Path("data/tiles/test_parallel/tiles32_RGBbands/val/23322E759967N_clipped_1m_1of2/labels_burned"),
            Path("data/tiles/test_parallel/tiles32_RGBbands/tst/23322E759967N_clipped_1m_2of2/labels_burned"),
        ]
        for labels_burned_dir in out_labels:
            # exact number may vary become of random sort between "trn" and "val"
            assert len(list(labels_burned_dir.iterdir())) > 25
        shutil.rmtree(Path(data_dir) / project_name)

    def test_tiling_inference(self):
        """Tests tiling of imagery only for inference"""
        data_dir = "data/tiles"
        project_name = "test_inference"
        cfg = {
            "general": {"project_name": project_name},
            "debug": True,
            "dataset": {
                "bands": ["R", "G", "B"],
                "raw_data_dir": data_dir,
                "raw_data_csv": "tests/tiling/tiling_segmentation_trn-inference_ci.csv",
                "attribute_field": "Quatreclasses",
                "attribute_values": [1, 2, 3, 4]
            },
            "tiling": {"tile_size": 32},
        }
        cfg = DictConfig(cfg)
        main(cfg)
        out_tiles = [
            (Path("data/tiles/test_inference/tiles32_RGBbands/trn/23322E759967N_clipped_1m_1of2/images"), 99),
            (Path("data/tiles/test_inference/tiles32_RGBbands/inference/23322E759967N_clipped_1m_2of2/images"), 176),
        ]
        for out_images_dir, num_tiles in out_tiles:
            assert out_images_dir.is_dir()
            assert len(list(out_images_dir.iterdir())) == num_tiles
        shutil.rmtree(Path(data_dir) / project_name)

    def test_annot_perc_crs_mismatch(self):
        # TODO: test annot_percent if bounds of imagery tile have different projection than gt tile
        pass
