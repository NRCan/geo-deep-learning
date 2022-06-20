from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
from shapely.geometry import box
from torchgeo.datasets.utils import extract_archive

from dataset.aoi import AOI
from utils.utils import read_csv
from utils.verifications import validate_features_from_gpkg


class Test_AOI(object):
    def test_multiband_input(self):
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])

    def test_singleband_input(self):
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['R', 'G', 'B'])

    def test_stac_input(self):
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-stac_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['red', 'green', 'blue'])

    def test_stac_url_input(self):
        extract_archive(src="tests/data/spacenet.zip")
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
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            row['gpkg'] = "missing_file.gpkg"
            with pytest.raises(AttributeError):
                aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])

    def test_bounds_iou(self) -> None:
        raster_file = "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif"
        raster = rasterio.open(raster_file)
        label_gdf = gpd.read_file('tests/data/massachusetts_buildings_kaggle/22978945_15.gpkg')

        label_bounds = label_gdf.total_bounds
        label_bounds_box = box(*label_bounds.tolist())
        raster_bounds_box = box(*list(raster.bounds))
        iou = AOI.bounds_iou(label_bounds_box, raster_bounds_box)
        expected_iou = 0.013904645827033404
        assert iou == expected_iou

    def test_parse_input_raster(self) -> None:
        extract_archive(src="tests/data/spacenet.zip")
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
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