import pytest
from torchgeo.datasets.utils import extract_archive

from models.model_choice import read_checkpoint
from utils.utils import read_csv, is_inference_compatible, update_gdl_checkpoint


class TestUtils(object):
    def test_wrong_seperation(self) -> None:
        extract_archive(src="tests/data/spacenet.zip")
        with pytest.raises(TypeError):
            data = read_csv("tests/tiling/point_virgule.csv")
        ##for row in data:
        ##aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])


    def test_with_header_in_csv(self) -> None:
        extract_archive(src="tests/data/spacenet.zip")
        with pytest.raises(TypeError):
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
