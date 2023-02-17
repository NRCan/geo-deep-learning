import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.crs import CRS
from torchgeo.datasets.utils import extract_archive
from torchgeo.datasets.utils import BoundingBox
from osgeo import gdal, gdalconst, ogr

from dataset.aoi import AOI
from tiling_segmentation import Tiler
from utils.geoutils import mask_nodata, nodata_vec_mask


class TestTiler(object):
    def test_tiling_per_aoi(self):
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
        img = "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif"
        gt = "tests/data/massachusetts_buildings_kaggle/22978945_15.gpkg"
        my_aoi = AOI(raster=img, raster_bands_request=[1, 2, 3], label=gt, split='trn')
        exp_dir = Path("tests/data/massachusetts_buildings_kaggle")
        tiling_dir = exp_dir / "patches"
        my_tiler = Tiler(
            tiling_root_dir=tiling_dir,
            src_aoi_list=[my_aoi],
            patch_size=32,
        )
        aoi, raster_patchs_paths, vect_patchs_paths = my_tiler.tiling_per_aoi(
            aoi=my_aoi,
            out_img_dir=tiling_dir / "images",
            out_label_dir=tiling_dir / "labels")
        assert len(raster_patchs_paths) == 15
        assert len(vect_patchs_paths) == 15
        assert Path(raster_patchs_paths[0]).is_file()
        assert Path(vect_patchs_paths[0]).is_file()
        shutil.rmtree(tiling_dir)

    def test_passes_min_annot(self):
        """Tests annotated percent calculation"""
        extract_archive(src="tests/data/spacenet.zip")
        img = "tests/data/spacenet/SN7_global_monthly_2020_01_mosaic_L15-0331E-1257N_1327_3160_13_uint8_clipped.tif"
        gt = "tests/data/spacenet/SN7_global_monthly_2020_01_mosaic_L15-0331E-1257N_1327_3160_13_uint8_clipped.gpkg"
        my_aoi = AOI(raster=img, raster_bands_request=[1, 2, 3], label=gt, split='trn')
        exp_dir = Path("tests/data/spacenet")
        tiling_dir = exp_dir / "patches"
        my_tiler = Tiler(
            tiling_root_dir=tiling_dir,
            src_aoi_list=[my_aoi],
        )
        for min_annot in range(10):
            my_tiler.min_annot_perc = min_annot
            passes, perc = my_tiler.passes_min_annot(img, gt)
            assert round(perc, 3) == 4.237
            if min_annot < perc:
                assert passes
            else:
                assert not passes
        shutil.rmtree(tiling_dir)

    def test_burn_gt_patch(self):
        """Tests burning a label while using the filter for attribute field and values"""
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        img = "tests/data/new_brunswick_aerial/23322E759967N_clipped_1m_1of2.tif"
        gt = "tests/data/new_brunswick_aerial/BakerLake_2017_clipped.gpkg"
        my_aoi = AOI(
            raster=img,
            raster_bands_request=[1, 2, 3],
            label=gt,
            split='trn',
            attr_field_filter="Quatreclasses",
            attr_values_filter=[4],
        )
        exp_dir = Path("tests/data/new_brunswick_aerial")
        tiling_dir = exp_dir / "patches"
        my_tiler = Tiler(
            tiling_root_dir=tiling_dir,
            src_aoi_list=[my_aoi],
        )
        gt_filtered = AOI.filter_gdf_by_attribute(
            gdf_patch=str(gt),
            attr_field=my_aoi.attr_field_filter,
            attr_vals=my_aoi.attr_values_filter
        )
        for continous_test, out_vals in zip([True, False], [[0, 1], [0, 4]]):
            out_vals_str = '-'.join([str(item) for item in out_vals])
            gt_patch_mask = tiling_dir / f"BakerLake_2017_clipped_mask_{out_vals_str}.tif"
            my_tiler.burn_gt_patch(
                aoi=my_aoi,
                img_patch=img,
                gt_patch=gt_filtered,
                out_px_mask=gt_patch_mask,
                continuous=continous_test,
                save_preview=False,
            )
            assert Path(gt_patch_mask).is_file()
            label_np = rasterio.open(gt_patch_mask).read()
            assert list(np.unique(label_np)) == out_vals
        shutil.rmtree(tiling_dir)

    def test__save_tile(self):
        """ Test _save_tile method of the Tiler class """
        sample = np.random.randint(0, 255, size=(4, 64, 32), dtype=np.uint8)
        dst_name = 'test_save_tile.tif'
        dst_dir = 'tests/data/test_save_tile'
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, dst_name)
        window = [230064.02203313413, 894213.4918, 230095.93486791675, 894181.5789652173]
        crs = 'PROJCS["NAD83 / Massachusetts Mainland",GEOGCS["NAD83",DATUM["North_American_Datum_1983",' \
              'SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],' \
              'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,' \
              'AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Lambert_Conformal_Conic_2SP"],' \
              'PARAMETER["latitude_of_origin",41],PARAMETER["central_meridian",-71.5],' \
              'PARAMETER["standard_parallel_1",42.6833333333333],PARAMETER["standard_parallel_2",41.7166666666667],' \
              'PARAMETER["false_easting",200000],PARAMETER["false_northing",750000],' \
              'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],' \
              'AUTHORITY["EPSG","26986"]]'

        Tiler._save_tile(sample=sample,
                         dst=dst,
                         window=window,
                         crs=crs
                         )

        # Calculate initial geotransforms:
        xmin, ymax, xmax, ymin = window
        n_rows, n_cols, n_bands = sample.shape[1], sample.shape[2], sample.shape[0]
        xres = (xmax - xmin) / float(n_cols)
        yres = (ymax - ymin) / float(n_rows)
        gt = (xmin, xres, 0, ymax, 0, -yres)

        # Open the saved patch and assert:
        ds = gdal.Open(dst, gdalconst.GA_ReadOnly)
        assert ds is not None, "The patch was not saved!"
        ds_arr = ds.ReadAsArray()
        assert (ds_arr == sample).all(), "Initial and saved arrays are different!"
        ds_crs = ds.GetProjection()
        assert ds_crs == crs, "Initial and saved CRSs are different!"
        ds_gt = ds.GetGeoTransform()
        assert ds_gt == gt, "Initial and saved geotransforms are different!"
        try:
            shutil.rmtree(dst_dir)
        except PermissionError:
            pass

    def test__parse_torchgeo_batch(self):
        """ Test _parse_torchgeo_batch method of the Tiler class """
        batch = {
            'image': np.random.randint(0, 255, size=(1, 3, 32, 32), dtype=np.uint8),
            'mask': [np.random.randint(0, 255, size=(1, 32, 32), dtype=np.uint8)],
            'crs': [CRS.from_epsg(26986)],
            'bbox': [BoundingBox(minx=230004.1545, maxx=230036.06733478262,
                                 miny=894044.8952545876, maxy=894076.8080893703, mint=0.0, maxt=9.223372036854776e+18)]


        }
        nodata = 0
        Tiler.for_inference = False
        sample_image, sample_mask, sample_crs, window = Tiler._parse_torchgeo_batch(Tiler, batch=batch, nodataval=nodata)

        assert (sample_image == batch['image']).all(), "Initial and unpacked samples images are different!"
        assert (sample_mask == batch['mask'][0]).all(), "Initial and unpacked samples masks are different!"
        assert CRS.from_wkt(sample_crs) == batch['crs'][0], "Initial and unpacked CRSs are different!"
        assert window[0] == batch['bbox'][0][0] and window[1] == batch['bbox'][0][3] and \
               window[2] == batch['bbox'][0][1] and window[3] == batch['bbox'][0][2],\
               "Initial and unpacked bboxes are different!"

        Tiler.for_inference = True
        sample_image, sample_mask, sample_crs, window = Tiler._parse_torchgeo_batch(Tiler, batch=batch, nodataval=nodata)
        assert (sample_image == batch['image']).all(), "Initial and unpacked samples images are different!"
        assert sample_mask is None, "The unpacked sample masks is not None! (Tiler.for_inference = True)"
        assert CRS.from_wkt(sample_crs) == batch['crs'][0], "Initial and unpacked CRSs are different!"
        assert window[0] == batch['bbox'][0][0] and window[1] == batch['bbox'][0][3] and \
               window[2] == batch['bbox'][0][1] and window[3] == batch['bbox'][0][2],\
               "Initial and unpacked bboxes are different!"

    def test__define_output_name(self):
        """ Test _define_output_name method of the Tiler class """
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
        img = "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif"
        gt = "tests/data/massachusetts_buildings_kaggle/22978945_15.gpkg"
        my_aoi = AOI(raster=img, raster_bands_request=[1, 2, 3], label=gt, split='trn')
        dst_dir = "tests/data"

        window = [1.1, 2.2, 3.3, 4.4]

        dst = Tiler._define_output_name(
            aoi=my_aoi,
            output_folder=dst_dir,
            window=window
        )

        assert dst == os.path.join(dst_dir, "22978945_15_uint8_clipped_1_1_2_2"), "Output file name does not " \
                                                                                  "match the input parameters!"

    def test__save_vec_mem_tile(self):
        """ Test _save_vec_mem_tile method of the Tiler class """
        """ Test _define_output_name method of the Tiler class """
        try:
            extract_archive(src="tests/data/massachusetts_buildings_kaggle_patch.zip")
        except FileNotFoundError:
            pass
        gt = "tests/data/massachusetts_buildings_kaggle_patch/massachusetts_buildings_kaggle_patch.gpkg"

        gt_ds = ogr.Open(gt)
        mem_driver = ogr.GetDriverByName('MEMORY')
        mem_gt_ds = mem_driver.CopyDataSource(gt_ds, 'mem_ds')

        dst_dir = 'tests/data/test__save_vec_mem_tile'
        dst_name = 'vec_ds.geojson'
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, dst_name)

        Tiler._save_vec_mem_tile(
            mem_ds=mem_gt_ds,
            output_vector_name=dst
        )

        saved_ds = ogr.Open(dst)
        assert saved_ds is not None, "The vector patch was not saved!"

        # Assert projections:
        gt_crs = gt_ds.GetLayer().GetSpatialRef().ExportToPrettyWkt()
        saved_crs = saved_ds.GetLayer().GetSpatialRef().ExportToPrettyWkt()
        assert gt_crs == saved_crs, "Initial and saved vector file projections are different!"

        # Assert bounding boxes:
        gt_bbox = gt_ds.GetLayer().GetExtent()
        saved_bbox = saved_ds.GetLayer().GetExtent()
        assert min([x - y < 0.000000001 for x, y in zip(gt_bbox, saved_bbox)]) == 1,\
               "Initial and saved vector file extents are different!"

        # Assert number of features:
        gt_fc = gt_ds.GetLayer().GetFeatureCount()
        saved_fc = saved_ds.GetLayer().GetFeatureCount()
        assert gt_fc == saved_fc, "Initial and saved numbers of features are different!"

        # Assert geometries:
        gt_layer = gt_ds.GetLayer()
        saved_layer = saved_ds.GetLayer()

        gt_layer.ResetReading()
        saved_layer.ResetReading()
        gt_feature = gt_layer.GetNextFeature()
        saved_feature = saved_layer.GetNextFeature()
        while gt_feature is not None:
            gt_geom = gt_feature.GetGeometryRef().GetEnvelope()
            saved_geom = saved_feature.GetGeometryRef().GetEnvelope()
            assert gt_geom == saved_geom, f"Initial and saved feature geometries are different!"
            gt_feature = gt_layer.GetNextFeature()
            saved_feature = saved_layer.GetNextFeature()
        try:
            shutil.rmtree(dst_dir)
        except PermissionError:
            pass

        del gt_ds
        del saved_ds

    def test_tiling_per_aoi_append_mode(self):
        """Tests tiling's append mode"""
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
        img = "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif"
        gt = "tests/data/massachusetts_buildings_kaggle/22978945_15.gpkg"
        my_aoi = AOI(raster=img, raster_bands_request=[1, 2, 3], label=gt, split='trn')
        exp_dir = Path("tests/data/massachusetts_buildings_kaggle")
        tiling_dir = exp_dir / "patches"
        my_tiler = Tiler(
            tiling_root_dir=tiling_dir,
            src_aoi_list=[my_aoi],
            patch_size=32,
            write_mode="raise_exists"
        )
        aoi, raster_patchs_paths, vect_patchs_paths = my_tiler.tiling_per_aoi(
            aoi=my_aoi,
            out_img_dir=tiling_dir / "images",
            out_label_dir=tiling_dir / "labels")
        raster_ctimes_init = {ras_patch: os.path.getctime(ras_patch) for ras_patch in raster_patchs_paths}
        vector_ctimes_init = {vec_patch: os.path.getctime(vec_patch) for vec_patch in vect_patchs_paths}
        # Rerun tiling in append mode
        my_tiler.write_mode = "append"
        aoi_skipped, raster_patchs_paths_skipped, vect_patchs_paths_skipped = my_tiler.tiling_per_aoi(
            aoi=my_aoi,
            out_img_dir=tiling_dir / "images",
            out_label_dir=tiling_dir / "labels")
        raster_ctimes_final = {ras_patch: os.path.getctime(ras_patch) for ras_patch in raster_patchs_paths_skipped}
        vector_ctimes_final = {vec_patch: os.path.getctime(vec_patch) for vec_patch in vect_patchs_paths_skipped}
        # check that no existing patches have been overwritten
        assert [ctimes for ctimes in raster_ctimes_init.values()] == [ctimes for ctimes in raster_ctimes_final.values()]
        assert [ctimes for ctimes in vector_ctimes_init.values()] == [ctimes for ctimes in vector_ctimes_final.values()]
        shutil.rmtree(tiling_dir)

    def test_mask_nodata(self):
        # Create a temporary directory to hold test files
        with TemporaryDirectory() as temp_dir:
            # Create test image and ground truth files
            image_path = Path(temp_dir) / 'test_image.tif'
            gt_path = Path(temp_dir) / 'test_gt.tif'
            image_arr = np.ones(shape=(3, 10, 10))
            image_arr[:, 3:6, 3:6] = 0
            gt_arr = np.zeros((10, 10))
            gt_arr[3:6, 3:6] = 1
            driver = gdal.GetDriverByName('GTiff')
            image_ds = driver.Create(str(image_path), 10, 10, 3, gdalconst.GDT_Byte)
            image_ds.GetRasterBand(1).WriteArray(image_arr[0, :, :])
            image_ds.GetRasterBand(2).WriteArray(image_arr[1, :, :])
            image_ds.GetRasterBand(3).WriteArray(image_arr[2, :, :])
            gt_ds = driver.Create(str(gt_path), 10, 10, 1, gdalconst.GDT_Byte)
            gt_ds.GetRasterBand(1).WriteArray(gt_arr)
            gt_ds = None
            image_ds = None

            # Call the function with test files and nodata value of 0
            mask_nodata(img_patch=image_path, gt_patch=gt_path, nodata_val=0, mask_val=255)

            # Check that nodata pixels in ground truth are masked
            gt_ds = gdal.Open(str(gt_path), gdalconst.GA_ReadOnly)
            masked_gt_arr = gt_ds.ReadAsArray()
            expected_masked_gt_arr = np.zeros((10, 10))
            expected_masked_gt_arr[3:6, 3:6] = 255
            assert (masked_gt_arr == expected_masked_gt_arr).all()
            gt_ds = None

    def test_nodata_vec_mask(self):
        with TemporaryDirectory() as tmp_path:
            # Create a temporary raster file
            img_path = Path(tmp_path) / 'test.tif'
            raster_drv = gdal.GetDriverByName('GTiff')
            raster = raster_drv.Create(str(img_path), 3, 3, 3, gdal.GDT_Byte)
            raster.SetProjection('EPSG:4326')
            raster.SetGeoTransform([0, 1, 0, 0, 0, -1])
            data = np.zeros(shape=(3, 3, 3), dtype=np.uint8)
            data[:, 1, 1] = 1
            raster.GetRasterBand(1).WriteArray(data[0, :, :])
            raster.GetRasterBand(1).SetNoDataValue(0)
            raster.GetRasterBand(2).WriteArray(data[1, :, :])
            raster.GetRasterBand(2).SetNoDataValue(0)
            raster.GetRasterBand(3).WriteArray(data[2, :, :])
            raster.GetRasterBand(3).SetNoDataValue(0)
            raster = None

            # Test nodata_vec_mask function
            with rasterio.open(img_path) as src:
                mask = nodata_vec_mask(raster=src)
                assert isinstance(mask, ogr.DataSource)
                layer = mask.GetLayer()
                feature = layer.GetFeature(0)
                geom = feature.GetGeometryRef()
                assert geom.GetGeometryName() == 'POLYGON'
                geom_wkt = geom.ExportToWkt()
                geom_wkt = geom_wkt.replace('POLYGON', '')
                geom_wkt = "".join(geom_wkt.split())
                assert geom_wkt == '((1-1,1-2,2-2,2-1,1-1))'

    def test_nodata_vec_mask_none(self):
        with TemporaryDirectory() as tmp_path:
            # Create a temporary raster file
            img_path = Path(tmp_path) / 'test.tif'
            raster_drv = gdal.GetDriverByName('GTiff')
            raster = raster_drv.Create(str(img_path), 3, 3, 3, gdal.GDT_Byte)
            raster.SetProjection('EPSG:4326')
            raster.SetGeoTransform([0, 1, 0, 0, 0, -1])
            data = np.ones(shape=(3, 3, 3), dtype=np.uint8)
            raster.GetRasterBand(1).WriteArray(data[0, :, :])
            raster.GetRasterBand(2).WriteArray(data[1, :, :])
            raster.GetRasterBand(3).WriteArray(data[2, :, :])
            raster = None

            # Test nodata_vec_mask function
            with rasterio.open(img_path) as src:
                mask = nodata_vec_mask(raster=src)
                assert mask is None

