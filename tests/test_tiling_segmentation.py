import itertools
import os
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

    def test_attribute_filter_continuous(self):
        """Tests the attribute field and values filter with continuous output"""
        data_dir = "data/tiles"
        Path(data_dir).mkdir(exist_ok=True)
        project_name = "test_attr_filter_cont"
        cfg = {
            "general": {"project_name": project_name},
            "debug": True,
            "dataset": {
                "bands": ["R", "G", "B"],
                "raw_data_dir": data_dir,
                "raw_data_csv": "tests/tiling/tiling_segmentation_multiclass_ci.csv",
                "attribute_field": "Quatreclasses",
                "attribute_values": [4],
                "min_annot_perc": 1
            },
            "tiling": None,
        }
        cfg = DictConfig(cfg)
        main(cfg)
        print(os.getcwd())
        assert Path(data_dir).is_dir()
        out_labels = list(Path(f"{data_dir}/{project_name}").glob("**/trn/**/labels_burned/*"))
        label = out_labels[0]
        label_np = rasterio.open(label).read()
        assert list(np.unique(label_np)) == [0, 1]
        shutil.rmtree(Path(data_dir) / project_name)

    def test_attribute_filter_discontinuous(self):
        """Tests the attribute field and values filter with discontinuous output pixel values"""
        data_dir = "data/tiles"
        Path(data_dir).mkdir(exist_ok=True)
        project_name = "test_attr_filter_discont"
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
        out_labels = list(Path(f"{data_dir}/{project_name}").glob("**/trn/**/labels_burned/*"))
        label = out_labels[0]
        label_np = rasterio.open(label).read()
        assert list(np.unique(label_np)) == [0, 4]
        shutil.rmtree(Path(data_dir) / project_name)

    def test_min_annotated_percent_filter(self):
        """Tests the minimum annotated percent filter"""
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        data = read_csv("tests/tiling/tiling_segmentation_multiclass_ci.csv")
        iterator = iter(data)
        row = next(iterator)
        annot_out = annot_percent(row['tif'], row['gpkg'])
        assert int(annot_out) == 330  # ground truth larger than raster

    def test_val_percent(self):
        """Tests the trn/val sorting to ensure the result is close enough to requested val_percent"""
        data_dir = f"data/tiles"
        Path(data_dir).mkdir(exist_ok=True)
        proj_prefix = "test_val_percent"
        datasets = {"binary-multiband", "multiclass"}
        results = []
        for expected_val_percent, min_annot, dataset in itertools.product([0.1, 0.2, 0.3, 0.4], [0, 1], datasets):
            proj_name = f"{proj_prefix}{str(expected_val_percent).replace('.', '')}_min_annot{min_annot}_{dataset}"
            cfg = {
                "general": {"project_name": proj_name},
                "debug": True,
                "dataset": {
                    "bands": ["R", "G", "B"],
                    "raw_data_dir": data_dir,
                    "raw_data_csv": f"tests/tiling/tiling_segmentation_{dataset}_ci.csv",
                    "train_val_percent": {'val': expected_val_percent},
                },
                "tiling": {
                    "tile_size": 32,
                    "min_annot_perc": min_annot
                },
            }
            cfg = DictConfig(cfg)
            main(cfg)
            out_labels_trn = list(Path(f"{data_dir}/{proj_name}/tiles32_RGBbands/trn").glob("*/labels_burned"))
            for out_lbls_aoi_trn in out_labels_trn:
                assert out_lbls_aoi_trn.is_dir()
                aoi_trn_parts = out_lbls_aoi_trn.parts
                out_labels_aoi_val = out_lbls_aoi_trn.parent.parent.parent / "val" / aoi_trn_parts[-2] / "labels_burned"
                assert out_labels_aoi_val.is_dir()
                out_trn_nb = len(list(out_lbls_aoi_trn.iterdir()))
                out_val_nb = len(list(out_labels_aoi_val.iterdir()))
                # some datasets, when filtering is applied, are just too small for this test to make sense
                actual_val_percent = out_val_nb/(out_trn_nb+out_val_nb)
                results.append((dataset, min_annot, expected_val_percent, actual_val_percent,
                                out_trn_nb, out_val_nb, out_trn_nb+out_val_nb))
                if out_trn_nb+out_val_nb < 100:  # allowed 70% tolerance for very small datasets
                    assert expected_val_percent*0.3 <= actual_val_percent <= expected_val_percent*1.7
                else:  # allowed 30% tolerance for very small datasets
                    assert expected_val_percent*0.7 <= actual_val_percent <= expected_val_percent*1.3
        for result in results:
            print(result)
        for dir in list(Path(data_dir).glob(f"{proj_prefix}*")):
            shutil.rmtree(dir)

    def test_annot_percent(self):
        """Tests the minimum annotated percentage to ensure the ground truth tiles with only background are rejected"""
        data_dir = f"data/tiles"
        Path(data_dir).mkdir(exist_ok=True)
        proj_prefix = "test_annot_percent"
        datasets = {"binary-multiband", "multiclass"}
        results = []
        for expected_min_annot, dataset in itertools.product([0, 1, 5, 10], datasets):
            proj_name = f"{proj_prefix}{expected_min_annot}_{dataset}"
            cfg = {
                "general": {"project_name": proj_name},
                "debug": True,
                "dataset": {
                    "bands": ["R", "G", "B"],
                    "raw_data_dir": data_dir,
                    "raw_data_csv": f"tests/tiling/tiling_segmentation_{dataset}_ci.csv",
                    "train_val_percent": {'val': 0.3},
                },
                "tiling": {
                    "tile_size": 32,
                    "min_annot_perc": expected_min_annot
                },
            }
            cfg = DictConfig(cfg)
            main(cfg)
            out_labels = list(Path(f"{data_dir}/{proj_name}/tiles32_RGBbands").glob("**/labels_burned/*"))
            for out_lbl in out_labels:
                assert out_lbl.is_file()
                out_lbl_rio = rasterio.open(out_lbl)
                out_lbl_np = out_lbl_rio.read()
                actual_annot_percent = out_lbl_np[out_lbl_np > 0].sum() / out_lbl_np.size * 100
                dataset = out_lbl.parts[4]
                if not dataset == 'tst':
                    assert dataset == 'trn' or dataset == 'val'
                    assert actual_annot_percent >= expected_min_annot * 0.99  # accept some tolerance
                results.append((out_lbl, dataset, expected_min_annot, actual_annot_percent))
        for result in results:
            print(result)
        for dir in list(Path(data_dir).glob(f"{proj_prefix}*")):
            shutil.rmtree(dir)

    def test_tiling_segmentation_parallel(self):
        data_dir = "data/tiles"
        Path(data_dir).mkdir(exist_ok=True)
        proj = "test_parallel"
        cfg = {
            "general": {"project_name": proj},
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
            (Path(f"{data_dir}/{proj}/tiles32_RGBbands/trn/23322E759967N_clipped_1m_1of2/labels_burned"), (80, 95)),
            (Path(f"{data_dir}/{proj}/tiles32_RGBbands/val/23322E759967N_clipped_1m_1of2/labels_burned"), (5, 20)),
            (Path(f"{data_dir}/{proj}/tiles32_RGBbands/tst/23322E759967N_clipped_1m_2of2/labels_burned"), (170, 190)),
        ]
        for labels_burned_dir, lbls_nb in out_labels:
            # exact number may vary because of random sort between "trn" and "val"
            assert lbls_nb[0] <= len(list(labels_burned_dir.iterdir())) <= lbls_nb[1]
        shutil.rmtree(Path(data_dir) / proj)

    def test_tiling_inference(self):
        """Tests tiling of imagery only for inference"""
        data_dir = "data/tiles"
        Path(data_dir).mkdir(exist_ok=True)
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
