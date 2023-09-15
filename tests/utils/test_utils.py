import multiprocessing
import os
import unittest
from pathlib import Path
from time import sleep

import pytest
import rasterio
from hydra.utils import to_absolute_path
from torchgeo.datasets.utils import extract_archive
from torchvision.datasets.utils import download_url

from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig

from models.model_choice import read_checkpoint
from utils.utils import read_csv, is_inference_compatible, update_gdl_checkpoint, get_key_def, download_url_wcheck, \
    map_wrapper
from utils.verifications import validate_raster


class TestUtils(unittest.TestCase):
    def test_wrong_seperation(self) -> None:
        extract_archive(src="tests/data/spacenet.zip")
        with pytest.raises(TypeError):
            data = read_csv("tests/tiling/point_virgule.csv")
        ##for row in data:
        ##aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])

    def test_with_header_in_csv(self) -> None:
        extract_archive(src="tests/data/spacenet.zip")
        with pytest.raises(ValueError):
            data = read_csv("tests/tiling/header.csv")
        ##for row in data:
        ##aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])

    def test_is_current_config(self) -> None:
        ckpt = "tests/utils/gdl_current_test.pth.tar"
        ckpt_dict = read_checkpoint(ckpt, update=False)
        assert is_inference_compatible(ckpt_dict)

    def test_update_gdl_checkpoint(self) -> None:
        ckpt = "tests/utils/gdl_pre20_test.pth.tar"
        ckpt_dict = read_checkpoint(ckpt, update=False)
        assert not is_inference_compatible(ckpt_dict)
        ckpt_updated = update_gdl_checkpoint(ckpt_dict)
        assert is_inference_compatible(ckpt_updated)

        # grouped to put emphasis on before/after result of updating
        assert ckpt_dict['params']['global']['number_of_bands'] == 4
        assert ckpt_updated['params']['dataset']['bands'] == ['red', 'green', 'blue', 'nir']

        assert ckpt_dict['params']['global']['num_classes'] == 1
        assert ckpt_updated['params']['dataset']['classes_dict'] == {'class1': 1}

        means = [0.0950882, 0.13039997, 0.12815733, 0.25175254]
        assert ckpt_dict['params']['training']['normalization']['mean'] == means
        assert ckpt_updated['params']['augmentation']['normalization']['mean'] == means

        assert ckpt_dict['params']['training']['augmentation']['clahe_enhance'] is True
        assert ckpt_updated['params']['augmentation']['clahe_enhance_clip_limit'] == 0.1

    def test_expected_type_get_key_def(self) -> None:
        with initialize(config_path="../../config", job_name="test_key_def"):
            cfg = compose(config_name="gdl_config_template", return_hydra_config=True)
            hconf = HydraConfig()
            hconf.set_config(cfg)
            # Not the same type - raise the error
            with self.assertRaises(TypeError):
                get_key_def('max_pix_per_mb_gpu', cfg['inference'], default=25, expected_type=str)
            # Same type
            mp = get_key_def('max_pix_per_mb_gpu', cfg['inference'], default=25, expected_type=int)
            assert isinstance(mp, int)
            # with to_path=True, two different possibilities for expected type: str or Path
            assert isinstance(cfg['inference']['raw_data_csv'], str)
            mp = get_key_def('raw_data_csv', cfg['inference'], expected_type=Path, to_path=True)
            assert isinstance(mp, Path)
            mp = get_key_def('raw_data_csv', cfg['inference'], expected_type=str, to_path=True)
            assert isinstance(mp, Path)

    def test_download_url_wcheck(self):
        """
        Tests parallel and simultaneous downloads of same file to make sure the second download doesn't consider file to
        be downloaded.
        """
        inputs = []
        # 94 MB
        url = "http://datacube-stage-data-public.s3.ca-central-1.amazonaws.com/store/imagery/optical/spacenet-samples/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-N.tif"
        filename = Path(url).name
        root = to_absolute_path("tests/utils")
        fpath = Path(root) / filename
        for i in range(2):
            init_sleep = 4 if i > 0 else 0
            inputs.append([download_and_validate, url, root, filename, init_sleep])
        with multiprocessing.get_context('spawn').Pool(None) as pool:
            pool.map_async(map_wrapper, inputs).get()
            os.remove(fpath)


def download_and_validate(url, root, filename, init_sleep, wcheck: bool = True):
    sleep(init_sleep)
    # basic download functions fails since multiple threads download same file
    if not wcheck:
        download_url(url, root, filename)
    else:
        # adapted function succeeds
        download_url_wcheck(url, root, filename)
    validate_raster(Path(root) / filename, extended=True)
