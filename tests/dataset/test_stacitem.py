import multiprocessing
import os.path
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pystac
import pytest
from _pytest.fixtures import SubRequest
import rasterio
from rasterio import RasterioIOError
from torchgeo.datasets.utils import extract_archive

from dataset.aoi import AOI, aois_from_csv
from dataset.stacitem import SingleBandItemEO
from utils.utils import read_csv


class Test_SingleBandItemEO(object):
    def test_stac_input_missing_band(self):
        """Tests error when requesting non-existing singleband input rasters from stac item"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-stac_ci.csv")
        row = next(iter(data))
        with pytest.raises(ValueError):
            item = SingleBandItemEO(item=pystac.Item.from_file(row['tif']),
                                    bands_requested=['ru', 'gris', 'but'])

    def test_stac_input_empty_band_request(self):
        """Tests error when band selection is required (stac item) but missing"""
        extract_archive(src="tests/data/spacenet.zip")
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
        stac_item_path = "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03.json"
        with pytest.raises(ValueError):
            item = SingleBandItemEO(item=pystac.Item.from_file(stac_item_path),
                                    bands_requested="")

    def test_is_valid_cname(self):
        assert SingleBandItemEO.is_valid_cname("nir")
        assert not SingleBandItemEO.is_valid_cname("N")
        assert not SingleBandItemEO.is_valid_cname(1)

    def test_band_to_cname(self):
        bands_invalid = [["R", "G", "B"], [1, 2, 3]]
        expected = ["red", "green", "blue"]
        for bands in bands_invalid:
            assert [SingleBandItemEO.band_to_cname(band) for band in bands] == expected
