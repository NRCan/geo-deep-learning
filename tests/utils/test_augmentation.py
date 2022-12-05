from typing import Dict

import pytest
import rasterio
import torch
from torchgeo.datasets.utils import extract_archive
from torchvision.utils import save_image

from utils.augmentation import compose_transforms
from utils.utils import read_csv


class TestAugmentation(object):
    @pytest.fixture(scope="class")
    def image(self):
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = data[0]
        sat_img = rasterio.open(row['tif']).read()
        sat_img = torch.from_numpy(sat_img / 255)
        return sat_img

    @pytest.fixture(scope="class")
    def params(self) -> Dict:
        cfg = {
            'augmentation': {
                'normalization': {
                    'mean': [0.1, 0.1, 0.1],
                    'std': [1.5, 1.5, 1.5]
                },
                'noise': 0,
                'hflip_prob': 1,
                'rotate_prob': 1,
                'rotate_limit': 90,
                'crop_size': 400,
            }
        }
        return cfg

    def test_compose_transforms_geom(self, params, image) -> None:
        mask = torch.zeros(image.shape)
        sample = {"image": image, "mask": mask}
        transforms = compose_transforms(params=params, dataset='trn')
        sample_transformed = transforms(sample)
        crop_size = params['augmentation']['crop_size']
        assert sample_transformed['image'].numpy().shape == (3, crop_size, crop_size)
        # for manual testing
        save_image((sample_transformed['image']), "tests/aug_test.tif")

    def test_compose_transforms_radiom(self, params, image) -> None:
        # Remove transforms with major impact on geometry
        params['augmentation']['crop_size'] = None
        params['augmentation']['rotate_limit'] = 0
        mask = torch.zeros(image.shape)
        sample = {"image": image, "mask": mask}
        transforms = compose_transforms(params=params, dataset='trn')
        sample_transformed = transforms(sample)
        norm_mean = params['augmentation']['normalization']['mean'][0]
        norm_std = params['augmentation']['normalization']['std'][0]
        # make sure normalization worked properly
        assert round((float(image.mean())-norm_mean)/norm_std, 2) == round(float(sample_transformed['image'].mean()), 2)
        # for manual testing
        # save_image((sample_transformed['image']), "tests/aug_test.tif")
