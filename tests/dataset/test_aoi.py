from pathlib import Path

import pytest

from dataset.aoi import AOI
from utils.utils import read_csv
from utils.verifications import validate_features_from_gpkg


class Test_AOI(object):
    def test_multiband_input(self):
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])

    def test_singleband_input(self):
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['R', 'G', 'B'])

    def test_stac_input(self):
        data = read_csv("tests/sampling/sampling_segmentation_binary-stac_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['red', 'green', 'blue'])

    def test_stac_url_input(self):
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband-url_ci.csv")
        for row in data:
            aoi = AOI(
                raster=row['tif'],
                label=row['gpkg'],
                split=row['split'],
                raster_bands_request=['R'],
                download_data=True,
                root_dir="data"
            )

    def test_missing_label(self):
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            row['gpkg'] = "missing_file.gpkg"
            with pytest.raises(AttributeError):
                aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])

    def test_parse_input_raster(self) -> None:
        raster_raw = {
            "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03.json": [
                "red", "green", "blue"],
            "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped_${dataset.bands}.tif": ["R", "G", "B"],
            "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif": None,
        }
        for raster_raw, bands_requested in raster_raw.items():
            raster_parsed = AOI.parse_input_raster(csv_raster_str=raster_raw, raster_bands_requested=bands_requested)
            print(raster_parsed)
# TODO: SingleBandItem
# test raise ValueError if request more than available bands