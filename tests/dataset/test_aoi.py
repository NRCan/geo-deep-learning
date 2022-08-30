import multiprocessing
import os.path
from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
from rasterio import RasterioIOError
from shapely.geometry import box
from torchgeo.datasets.utils import extract_archive

from dataset.aoi import AOI
from utils.utils import read_csv


class Test_AOI(object):
    def test_multiband_input(self):
        """Tests reading a multiband raster as input"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
            aoi.close_raster()

    def test_singleband_input(self):
        """Tests reading a singleband raster as input with ${dataset.bands} pattern"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['R', 'G', 'B'])
            aoi.close_raster()

    def test_stac_input(self):
        """Tests singleband raster referenced by stac item as input"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-stac_ci.csv")
        for row in data:
            aoi = AOI(
                raster=row['tif'], label=row['gpkg'], split=row['split'],
                raster_bands_request=['red', 'green', 'blue'])
            aoi.close_raster()

    def test_stac_url_input(self):
        """Tests download of singleband raster as url path referenced by a stac item"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband-url_ci.csv")
        for row in data:
            aoi = AOI(
                raster=row['tif'], label=row['gpkg'], split=row['split'],
               raster_bands_request=['R'], download_data=True, root_dir="data"
            )
            assert aoi.download_data == True
            assert Path("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-R.tif").is_file()
            aoi.close_raster()
            os.remove("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-R.tif")

    def test_missing_label(self):
        """Tests error when missing label file"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            row['gpkg'] = "missing_file.gpkg"
            with pytest.raises(AttributeError):
                aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
                aoi.close_raster()

    def test_parse_input_raster(self) -> None:
        """Tests parsing for three accepted patterns to reference input raster data with band selection"""
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

    def test_bounds_iou(self) -> None:
        """Tests calculation of IOU between raster and label bounds"""
        raster_file = "tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped.tif"
        raster = rasterio.open(raster_file)
        label_gdf = gpd.read_file('tests/data/massachusetts_buildings_kaggle/22978945_15.gpkg')

        label_bounds = label_gdf.total_bounds
        label_bounds_box = box(*label_bounds.tolist())
        raster_bounds_box = box(*list(raster.bounds))
        iou = AOI.bounds_iou(label_bounds_box, raster_bounds_box)
        expected_iou = 0.013904645827033404
        assert iou == expected_iou

    def test_corrupt_raster(self) -> None:
        """Tests error when reading a corrupt file"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            row['tif'] = "tests/data/massachusetts_buildings_kaggle/corrupt_file.tif"
            with pytest.raises(BaseException):
                aoi = AOI(raster=row['tif'], label=None)
                aoi.close_raster()

    def test_image_only(self) -> None:
        """Tests AOI creation with image only, ie no label"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=None)
            assert aoi.label is None
            aoi.close_raster()

    def test_missing_raster(self) -> None:
        """Tests error when pointing to missing raster"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            row['tif'] = "missing_raster.tif"
            with pytest.raises(RasterioIOError):
                aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
                aoi.close_raster()

    def test_wrong_split(self) -> None:
        """Tests error when setting a wrong split, ie not 'trn', 'tst' or 'inference'"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            row['split'] = "missing_split"
            with pytest.raises(ValueError):
                aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
                aoi.close_raster()

    def test_download_data(self) -> None:
        """Tests download data"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            row['tif'] = "http://datacube-stage-data-public.s3.ca-central-1.amazonaws.com/store/imagery/optical/" \
                         "spacenet-samples/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-N.tif"
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], download_data=True, root_dir="data")
            assert Path("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-N.tif").is_file()
            assert aoi.download_data is True
            aoi.close_raster()
            os.remove("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-N.tif")

    def test_stac_input_missing_band(self):
        """Tests error when requestinga non-existing singleband input rasters from stac item"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-stac_ci.csv")
        for row in data:
            with pytest.raises(ValueError):
                aoi = AOI(
                    raster=row['tif'], label=row['gpkg'], split=row['split'],
                    raster_bands_request=['ru', 'gris', 'but'])
                aoi.close_raster()

    def test_no_intersection(self) -> None:
        """Tests error testing no intersection between raster and label"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            row['gpkg'] = "tests/data/new_brunswick_aerial/BakerLake_2017_clipped.gpkg"
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
            assert aoi.bounds_iou == 0
            aoi.close_raster()

    def test_write_multiband_from_single_band(self) -> None:
        """Tests the 'write_multiband' method"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['R', 'G', 'B'],
                      write_multiband=True, root_dir="data")
            assert Path("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB.tif").is_file()
            aoi.close_raster()
            os.remove("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB.tif")

    def test_write_multiband_from_single_band_url(self) -> None:
        """Tests the 'write_multiband' method with singleband raster as URL"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband-url_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['R', 'G', 'B'],
                      write_multiband=True, root_dir="data", download_data=True)
            assert Path("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB.tif").is_file()
            assert aoi.download_data == True
            aoi.close_raster()
            for bands in ["RGB", "R", "G", "B"]:
                os.remove(f"data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-{bands}.tif")

    def test_download_true_not_url(self) -> None:
        """Tests AOI creation if download_data set to True, but not necessary (local image)"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], download_data=True,
                      raster_bands_request=['R', 'G', 'B'])
            aoi.close_raster()

    def test_raster_stats_from_stac(self) -> None:
        """Tests the calculation of statistics of raster data as stac item from an AOI instance"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-stac_ci.csv")
        bands_request = ['red', 'green', 'blue']
        expected_stats = {
            'red': {'statistics': {'minimum': 0, 'maximum': 255, 'mean': 10.133578590682399, 'median': 9.0,
                                   'std': 9.657404407462819}},
            'green': {'statistics': {'minimum': 0, 'maximum': 255, 'mean': 19.382699380973825, 'median': 20.0,
                                     'std': 14.121321954317324}},
            'blue': {'statistics': {'minimum': 0, 'maximum': 255, 'mean': 21.429019052130965, 'median': 24.0,
                                    'std': 13.91999326196311}},
            'all': {'statistics': {'minimum': 0.0, 'maximum': 255.0, 'mean': 16.98176567459573,
                                   'median': 17.666666666666668, 'std': 12.566239874581086}}}
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=bands_request)
            stats = aoi.calc_raster_stats()
            for band, band_stat in stats.items():
                assert band_stat['statistics'] == expected_stats[band]['statistics']
                assert len(band_stat['histogram']['buckets']) == 256
            aoi.close_raster()
            break

    def test_raster_stats_not_stac(self) -> None:
        """Tests the calculation of statistics of local multiband raster data from an AOI instance"""
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        data = read_csv("tests/sampling/sampling_segmentation_multiclass_ci.csv")
        expected_stats = {
            'band_0': {'statistics': {'minimum': 11, 'maximum': 254, 'mean': 159.36075617930456, 'median': 165.0,
                                      'std': 48.9924913616138}},
            'band_1': {'statistics': {'minimum': 12, 'maximum': 255, 'mean': 149.58768328445748, 'median': 154.0,
                                      'std': 46.204003828563714}},
            'band_2': {'statistics': {'minimum': 26, 'maximum': 254, 'mean': 119.60827398408044, 'median': 117.0,
                                      'std': 41.85516710316288}},
            'all': {'statistics': {'minimum': 16.333333333333332, 'maximum': 254.33333333333334,
                                   'mean': 142.8522378159475, 'median': 145.33333333333334, 'std': 45.68388743111347}}}
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
            stats = aoi.calc_raster_stats()
            for band, band_stat in stats.items():
                assert band_stat['statistics'] == expected_stats[band]['statistics']
                assert len(band_stat['histogram']['buckets']) == 256
            aoi.close_raster()
            break

    def test_to_dict(self):
        """Test the 'to_dict()' method on an AOI instance"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-stac_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['red', 'green', 'blue'])
            aoi.to_dict()
            aoi.close_raster()

    def test_for_multiprocessing(self) -> None:
        """Tests multiprocessing on AOI instances"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_multiclass_ci.csv")
        inputs = []
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], for_multiprocessing=True)
            inputs.append([aoi_read_raster, aoi])
            assert aoi.raster_closed is True
            assert aoi.raster is None

        with multiprocessing.get_context('spawn').Pool(None) as pool:
            aoi_meta = pool.map_async(map_wrapper, inputs).get()
        print(aoi_meta)


def map_wrapper(x):
    """For multi-threading"""
    return x[0](*(x[1:]))


def aoi_read_raster(aoi: AOI):
    """Function to package in multiprocessing"""
    aoi.raster_read()
    return aoi.raster.meta

