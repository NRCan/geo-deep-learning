from typing import List
from tempfile import NamedTemporaryFile

import pytest
import rasterio
from rasterio.io import DatasetReader
from rasterio.crs import CRS
from torchgeo.datasets.utils import extract_archive
from torchgeo.datasets.utils import BoundingBox
from osgeo import ogr
from _pytest.fixtures import SubRequest
import torch

from dataset.create_dataset import DRDataset, GDLVectorDataset


class TestDRDataset:
    @pytest.fixture(params=["tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif",
                            "tests/data/massachusetts_buildings_kaggle/23429155_15_uint8_clipped.tif"]
                    )
    def raster_dataset(self, request: SubRequest) -> DatasetReader:
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
        image = request.param
        dr_ds = rasterio.open(image)
        return dr_ds

    def test_getitem_single_patch(self, raster_dataset: DatasetReader) -> None:
        ds = DRDataset(raster_dataset)
        x = ds[ds.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert raster_dataset.read().shape[0] == x["image"].shape[0]
        assert isinstance(x['bbox'], BoundingBox)


class TestGDLVectorDataset:
    def test_init(self):
        # Test initialization with valid data
        vec_ds = NamedTemporaryFile(suffix=".gpkg").name
        ds = ogr.GetDriverByName('GPKG').CreateDataSource(vec_ds)
        layer = ds.CreateLayer("test", geom_type=ogr.wkbPolygon)
        ds = None

        gdl_vec_ds = GDLVectorDataset(vec_ds=vec_ds)

        assert isinstance(gdl_vec_ds, GDLVectorDataset)
        assert isinstance(gdl_vec_ds.ds, ogr.DataSource)
        assert gdl_vec_ds.vec_srs is not None
        assert gdl_vec_ds._crs is not None

        # Test initialization with empty data
        with pytest.raises(AssertionError):
            GDLVectorDataset(vec_ds=None)

    @pytest.fixture(params=["tests/data/massachusetts_buildings_kaggle/22978945_15.gpkg",
                            "tests/data/massachusetts_buildings_kaggle/23429155_15.gpkg"]
                    )
    def vector_dataset(self, request: SubRequest) -> GDLVectorDataset:
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
        fp = request.param
        vec_ds = GDLVectorDataset(fp)
        return vec_ds

    def test_getitem(self, vector_dataset: GDLVectorDataset) -> None:
        x = vector_dataset[vector_dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], ogr.DataSource)
        assert isinstance(x['bbox'], BoundingBox)

    def test_clip_nodata(self):
        # Create a vector dataset:
        vec_ds_name = NamedTemporaryFile(suffix=".gpkg").name
        ds = ogr.GetDriverByName('GPKG').CreateDataSource(vec_ds_name)
        layer = ds.CreateLayer("test", geom_type=ogr.wkbPolygon)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkt('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'))
        layer.CreateFeature(feature)
        ds = None

        nodata_mask = ogr.GetDriverByName('MEMORY').CreateDataSource('maskdata')
        layer = nodata_mask.CreateLayer('0', geom_type=ogr.wkbPolygon)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkt('POLYGON ((0.25 0.25, 0.75 0.25, 0.75 0.75, 0.25 0.75, 0.25 0.25))'))
        layer.CreateFeature(feature)
        feature = None

        gdl_vec_ds = GDLVectorDataset(vec_ds=vec_ds_name)
        clipped_ds = gdl_vec_ds._clip_nodata(gdl_vec_ds.ds, nodata_mask)

        assert isinstance(clipped_ds, ogr.DataSource)
        layer = clipped_ds.GetLayer()
        feature = layer.GetFeature(0)
        geom = feature.GetGeometryRef()
        assert geom.GetGeometryName() == 'POLYGON'
        geom_wkt = geom.ExportToWkt()
        geom_wkt = geom_wkt.replace('POLYGON ', '')
        geom_wkt = geom_wkt.replace('((', '')
        geom_wkt = geom_wkt.replace('))', '')
        geom_wkt = set(geom_wkt.split(","))
        assert geom_wkt == set(['0.25 0.25', '0.75 0.25', '0.75 0.75', '0.25 0.75', '0.25 0.25'])

        # Test clipping without nodata mask
        nodata_mask = None
        clipped_ds = gdl_vec_ds._clip_nodata(gdl_vec_ds.ds, nodata_mask)

        assert isinstance(clipped_ds, ogr.DataSource)
        assert clipped_ds == gdl_vec_ds.ds


class TestIntersectionCustomDatasets:
    @pytest.fixture(params=zip(["tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif",
                                "tests/data/massachusetts_buildings_kaggle/23429155_15_uint8_clipped.tif"],
                               ["tests/data/massachusetts_buildings_kaggle/22978945_15.gpkg",
                                "tests/data/massachusetts_buildings_kaggle/23429155_15.gpkg"]
                               )
                    )
    def dataset(self, request: SubRequest) -> List:
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
        image = request.param[0]
        dr_ds = rasterio.open(image)
        raster_ds = DRDataset(dr_ds)
        fp = request.param[1]
        vec_ds = GDLVectorDataset(fp)
        intersection_ds = raster_ds & vec_ds
        return [intersection_ds, raster_ds, vec_ds]

    def test_getitem(self, dataset:  List) -> None:
        x = dataset[0][dataset[0].bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], ogr.DataSource)
        assert isinstance(x['bbox'], BoundingBox)

    def test_intersection_bbox(self, dataset: List) -> None:
        intersection_bounds = dataset[0].bounds
        raster_bbox = dataset[1].bounds
        vector_bbox = dataset[2].bounds

        minx = max(raster_bbox[0], vector_bbox[0])
        maxx = min(raster_bbox[1], vector_bbox[1])
        miny = max(raster_bbox[2], vector_bbox[2])
        maxy = min(raster_bbox[3], vector_bbox[3])
        assert intersection_bounds[0] == minx
        assert intersection_bounds[1] == maxx
        assert intersection_bounds[2] == miny
        assert intersection_bounds[3] == maxy
