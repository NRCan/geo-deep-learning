import itertools
import shutil
from pathlib import Path

import numpy
import pytest
import rasterio
from omegaconf import DictConfig
from torchgeo.datasets.utils import extract_archive

from tiling_segmentation import annot_percent
from tiling_segmentation import main as tiling
from utils.utils import read_csv


class TestTiling(object):
    def test_outputted_chips(self):
        """Tests the tiling process to ensure all expected tiles are created"""
        data_dir = f"data/patches"
        proj = f"tiling_output_test"
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
        cfg = {
            "general": {"project_name": proj},
            "debug": True,
            "dataset": {
                "bands": [1, 2, 3],
                "raw_data_dir": data_dir,
                "raw_data_csv": f"tests/tiling/tiling_segmentation_binary_ci.csv",
            },
            "tiling": {
                "patch_size": 32,
                "train_val_percent": {'val': 0.3},
            },
        }
        cfg = DictConfig(cfg)
        tiling(cfg)
        # expected number of patches is constant due to random seed set in tiling script
        out_labels = [
            (Path(f"{data_dir}/{proj}/trn/22978945_15_uint8_clipped/labels_burned"), 13),
            (Path(f"{data_dir}/{proj}/val/22978945_15_uint8_clipped/labels_burned"), 2),
            (Path(f"{data_dir}/{proj}/tst/23429155_15_uint8_clipped/labels_burned"), 9),
        ]
        for dataset, out in zip(["trn", "val", "tst"], out_labels):
            number_labels_actual = len(list(out[0].iterdir()))
            number_labels_expected = out[-1]
            assert number_labels_actual == number_labels_expected
        shutil.rmtree(Path(data_dir) / proj)

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
        data_dir = f"data/patches"
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        extract_archive(src="tests/data/spacenet.zip")
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        proj_prefix = "test_val_percent"
        datasets = {"binary-multiband", "multiclass"}
        results = []
        for expected_val_percent, min_annot, dataset in itertools.product([0.1, 0.4], [0, 1], datasets):
            proj_name = f"{proj_prefix}{str(expected_val_percent).replace('.', '')}_min_annot{min_annot}_{dataset}"
            cfg = {
                "general": {"project_name": proj_name},
                "debug": True,
                "dataset": {
                    "bands": [1, 2, 3],
                    "raw_data_dir": data_dir,
                    "raw_data_csv": f"tests/tiling/tiling_segmentation_{dataset}_ci.csv",
                },
                "tiling": {
                    "patch_size": 32,
                    "min_annot_perc": min_annot,
                    "train_val_percent": {'val': expected_val_percent},
                },
            }
            cfg = DictConfig(cfg)
            tiling(cfg)
            out_labels_trn = list(Path(f"{data_dir}/{proj_name}/trn").glob("*/labels_burned"))
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
        """Tests the minimum annotated percentage to assert ground truth patches with mostly background are rejected"""
        data_dir = f"data/patches"
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        extract_archive(src="tests/data/spacenet.zip")
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        proj_prefix = "test_annot_percent"
        datasets = {"binary-multiband", "multiclass"}
        results = []
        for expected_min_annot, dataset in itertools.product([0, 5, 10], datasets):
            proj_name = f"{proj_prefix}{expected_min_annot}_{dataset}"
            cfg = {
                "general": {"project_name": proj_name},
                "debug": True,
                "dataset": {
                    "bands": [1, 2, 3],
                    "raw_data_dir": data_dir,
                    "raw_data_csv": f"tests/tiling/tiling_segmentation_{dataset}_ci.csv",
                },
                "tiling": {
                    "patch_size": 32,
                    "min_annot_perc": expected_min_annot,
                    "train_val_percent": {'val': 0.3},
                },
            }
            cfg = DictConfig(cfg)
            tiling(cfg)
            out_labels = list(Path(f"{data_dir}/{proj_name}").glob("*/*/labels_burned/*"))
            for out_lbl in out_labels:
                assert out_lbl.is_file()
                out_lbl_rio = rasterio.open(out_lbl)
                out_lbl_np = out_lbl_rio.read()
                actual_annot_percent = out_lbl_np[out_lbl_np > 0].sum() / out_lbl_np.size * 100
                dataset = out_lbl.parts[3]
                if not dataset == 'tst':
                    assert dataset == 'trn' or dataset == 'val'
                    assert actual_annot_percent >= expected_min_annot * 0.99  # accept some tolerance
                results.append((out_lbl, dataset, expected_min_annot, actual_annot_percent))
        for result in results:
            print(result)
        for dir in list(Path(data_dir).glob(f"{proj_prefix}*")):
            try:
                shutil.rmtree(dir)
            except PermissionError:
                pass

    def test_tiling_segmentation_parallel(self):
        data_dir = "data/patches"
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        proj = "test_parallel"
        cfg = {
            "general": {"project_name": proj},
            "debug": True,
            "dataset": {
                "bands": [1, 2, 3],
                "raw_data_dir": data_dir,
                "raw_data_csv": "tests/tiling/tiling_segmentation_multiclass_ci.csv",
                "attribute_field": "Quatreclasses",
                "attribute_values": [1, 2, 3, 4]
            },
            "tiling": {
                "multiprocessing": True,
                "patch_size": 32,
            }
        }
        cfg = DictConfig(cfg)
        tiling(cfg)
        out_labels = [
            (Path(f"{data_dir}/{proj}/trn/23322E759967N_clipped_1m_1of2/labels_burned"), (80, 95)),
            (Path(f"{data_dir}/{proj}/val/23322E759967N_clipped_1m_1of2/labels_burned"), (5, 20)),
            (Path(f"{data_dir}/{proj}/tst/23322E759967N_clipped_1m_2of2/labels_burned"), (170, 190)),
        ]
        for labels_burned_dir, lbls_nb in out_labels:
            # exact number may vary because of random sort between "trn" and "val"
            assert lbls_nb[0] <= len(list(labels_burned_dir.iterdir())) <= lbls_nb[1]
        shutil.rmtree(Path(data_dir) / proj)

    def test_tiling_inference(self):
        """Tests tiling of imagery only for inference"""
        data_dir = "data/patches"
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        project_name = "test_inference"
        cfg = {
            "general": {"project_name": project_name},
            "debug": True,
            "dataset": {
                "bands": [1, 2, 3],
                "raw_data_dir": data_dir,
                "raw_data_csv": "tests/tiling/tiling_segmentation_trn-inference_ci.csv",
                "attribute_field": "Quatreclasses",
                "attribute_values": [1, 2, 3, 4]
            },
            "tiling": {"patch_size": 32},
        }
        cfg = DictConfig(cfg)
        tiling(cfg)
        out_patches = [
            (Path("data/patches/test_inference/trn/23322E759967N_clipped_1m_1of2/images"), 99),
            (Path("data/patches/test_inference/inference/23322E759967N_clipped_1m_2of2/images"), 176),
        ]
        for out_images_dir, num_patches in out_patches:
            assert out_images_dir.is_dir()
            assert len(list(out_images_dir.iterdir())) == num_patches
        shutil.rmtree(Path(data_dir) / project_name)

    def test_annot_perc_crs_mismatch(self):
        """Tests annotated percent calculation if bounds of imagery have different projection than ground truth"""
        img = "tests/data/spacenet/SN7_global_monthly_2020_01_mosaic_L15-0331E-1257N_1327_3160_13_uint8_clipped.tif"
        gt = "tests/data/spacenet/SN7_global_monthly_2020_01_mosaic_L15-0331E-1257N_1327_3160_13_uint8_clipped_4326.gpkg"
        with pytest.raises(rasterio.errors.CRSError):
            perc = annot_percent(img, gt)

    def test_tiling_equalization(self):
        """Tests the tiling process with clahe equalization"""
        means_per_limit = []
        for clip_limit in [0, 10, 25]:
            data_dir = f"data/patches"
            proj = f"tiling_output_test"
            Path(data_dir).mkdir(exist_ok=True, parents=True)
            extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
            cfg = {
                "general": {"project_name": proj},
                "debug": True,
                "dataset": {
                    "bands": [1, 2, 3],
                    "raw_data_dir": data_dir,
                    "raw_data_csv": f"tests/tiling/tiling_segmentation_binary_ci.csv",
                },
                "tiling": {
                    "patch_size": 32,
                    "train_val_percent": {'val': 0.3},
                    "clahe_clip_limit": clip_limit,
                },
            }
            cfg = DictConfig(cfg)
            tiling(cfg)
            out_labels = (Path(data_dir)/proj).glob("**/images/*.tif")
            patch_means = []
            for patch in out_labels:
                patch_np = rasterio.open(patch).read()
                patch_means.append(patch_np)
            aoi_means = int(numpy.asarray(patch_means).mean())
            means_per_limit.append(aoi_means)
        assert means_per_limit[0] < means_per_limit[1] < means_per_limit[2]
        shutil.rmtree(Path(data_dir) / proj)
