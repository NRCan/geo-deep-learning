import multiprocessing
from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
from shapely.geometry import box
from torchgeo.datasets.utils import extract_archive

from dataset.aoi import AOI
from utils.utils import read_csv

class Test_utils(object):
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