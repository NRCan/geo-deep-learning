import pytest
from torchgeo.datasets.utils import extract_archive

from utils.utils import read_csv

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