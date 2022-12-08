import os
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from _pytest.fixtures import SubRequest
from torchgeo.datasets.utils import extract_archive

from dataset.aoi import AOI
from utils.geoutils import create_new_raster_from_base, bounds_gdf, bounds_riodataset, overlap_poly1_rto_poly2
from utils.utils import read_csv


class TestGeoutils(object):
    def test_multiband_vrt_from_single_band(self) -> None:
        """Tests the 'stack_singlebands_vrt' utility"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-singleband_ci.csv")
        row = data[0]
        bands_request = ['R', 'G', 'B']
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'],
                  raster_bands_request=bands_request, root_dir="data")
        assert aoi.raster.count == len(bands_request)
        src_red = rasterio.open(aoi.raster_raw_input.replace("${dataset.bands}", "R"))
        src_red_np = src_red.read()
        dest_red_np = aoi.raster.read(1)
        # make sure first band in multiband VRT is identical to source R band
        assert np.all(src_red_np == dest_red_np)
        aoi.close_raster()

    @pytest.fixture(
        params=[[1], [1, 2], [1, 2, 3]]
    )
    def bands_request(self, request: SubRequest) -> List:
        return request.param

    def test_create_new_raster_from_base_bands(self, bands_request) -> None:
        """Tests the 'create_new_raster_from_base' geo-utility for different output bands number"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        ref_raster = Path(data[0]['tif'])
        out_raster = ref_raster.parent / f"{ref_raster.stem}_copy.tif"
        out_array = rasterio.open(ref_raster).read(bands_request)
        create_new_raster_from_base(input_raster=ref_raster, output_raster=out_raster, write_array=out_array)
        os.remove(out_raster)

    def test_bounds_iou(self) -> None:
        """Tests calculation of IOU between raster and label bounds"""
        raster_file = "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif"
        raster = rasterio.open(raster_file)
        label_gdf = gpd.read_file('tests/data/massachusetts_buildings_kaggle/22978945_15.gpkg')

        label_bounds_box = bounds_gdf(label_gdf)
        raster_bounds_box = bounds_riodataset(raster)
        overlap_label_rto_raster = overlap_poly1_rto_poly2(label_bounds_box, raster_bounds_box)
        overlap_raster_rto_label = overlap_poly1_rto_poly2(raster_bounds_box, label_bounds_box)
        expected_overlap_lab_rto_ras = 1.0
        expected_overlap_ras_rto_lab = 0.014
        assert round(overlap_label_rto_raster, 3) == expected_overlap_lab_rto_ras
        assert round(overlap_raster_rto_label, 3) == expected_overlap_ras_rto_lab

    def test_empty_geopackage_overlap(self):
        """ Tests calculation of overlap of raster relative to an empty geopackage """
        extract_archive(src="tests/data/buil_AB11-WV02-20100926-1.zip")
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
        raster_file = "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif"
        raster = rasterio.open(raster_file)
        label_gdf = gpd.read_file('tests/data/buil_AB11-WV02-20100926-1.gpkg')
        label_bounds_box = bounds_gdf(label_gdf)
        raster_bounds_box = bounds_riodataset(raster)
        assert overlap_poly1_rto_poly2(label_bounds_box, raster_bounds_box) == 0.0
