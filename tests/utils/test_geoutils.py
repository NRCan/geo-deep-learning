import os
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from _pytest.fixtures import SubRequest
from torchgeo.datasets.utils import extract_archive
from shapely.geometry import MultiPolygon

from dataset.aoi import AOI
from utils.geoutils import create_new_raster_from_base, bounds_gdf, bounds_riodataset, overlap_poly1_rto_poly2, \
    multi2poly, fetch_tag_raster, gdf_mean_vertices_nb, check_gdf_load
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

    def test_create_new_raster_from_base_shape(self) -> None:
        """
        Tests error in 'create_new_raster_from_base' geo-utility if output array dimensions is not consistant with input
        raster
        """
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        ref_raster = Path(data[0]['tif'])
        out_raster = ref_raster.parent / f"{ref_raster.stem}_copy.tif"
        out_array = rasterio.open(ref_raster).read()[..., :20]  # read only part of the original width
        with pytest.raises(ValueError):
            create_new_raster_from_base(input_raster=ref_raster, output_raster=out_raster, write_array=out_array)

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
        
    def test_fetch_tag_raster(self):
        """Test to verify if the function really fetch the right information from the raster"""
        raster_file = "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif"
        tag_raster = 'tests/data/massachusetts_buildings_kaggle/tag_raster.tif'
        with rasterio.open(raster_file, 'r') as src_ds:
            with rasterio.open(tag_raster, 'w', **src_ds.meta) as dst_ds:
                dst_ds.update_tags(checkpoint='test/path/to/checkpoint.pth')
                dst_ds.write(src_ds.read())
        assert fetch_tag_raster(tag_raster, 'checkpoint') == 'test/path/to/checkpoint.pth'
         
    def test_multi2poly(self):
        """Test the conversion from MultiPolygon to Polygon in a GPKG"""
        multi_gpkg = "tests/data/new_brunswick_aerial/BakerLake_2017_clipped.gpkg"
        multi2poly(multi_gpkg, 'BakerLake_2017')
        df = gpd.read_file(multi_gpkg, layer='BakerLake_2017')
        have_multi = 'MultiPolygon' in df['geometry'].geom_type.values
        assert have_multi == False

    def test_gdf_mean_vertices_nb(self):
        test_data_labels_csv = "tests/data/all_gpkg.csv"
        with open(test_data_labels_csv, 'r') as fh:
            test_data_labels = fh.read().splitlines()
        mean_vertices_per_label = []
        for label in test_data_labels:
            my_gdf = check_gdf_load(label)
            mean_vertices = gdf_mean_vertices_nb(my_gdf)
            mean_vertices_per_label.append(mean_vertices)
        mean_vertices_per_label_int = [round(mean_verts) for mean_verts in mean_vertices_per_label if mean_verts]
        assert mean_vertices_per_label_int == [7, 7, 6, 36, 5, 5, 8, 5]
