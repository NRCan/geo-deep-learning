import multiprocessing
import os.path
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio import RasterioIOError
from shapely.geometry import box
from torchgeo.datasets.utils import extract_archive

from dataset.aoi import AOI, aois_from_csv
from utils.utils import read_csv


class Test_AOI(object):
    def test_multiband_input(self):
        """Tests reading a multiband raster as input"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = data[0]
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
        assert aoi.raster_raw_input == row['tif']
        assert aoi.split == row['split']
        assert str(aoi.label) == row['gpkg']
        assert aoi.aoi_id == 'SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB'
        src_count = rasterio.open(aoi.raster_raw_input).count
        assert src_count == aoi.raster.count
        assert str(Path("tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB.tif")) in \
               str(aoi.raster_name)
        assert isinstance(aoi.label_gdf, gpd.GeoDataFrame) and not aoi.label_gdf.empty
        assert not aoi.raster_closed
        aoi.close_raster()
        assert aoi.raster_closed

    def test_multiband_input_band_selection(self):
        """Tests reading a multiband raster as input with band selection"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = data[0]
        bands_request = [2, 1]
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=bands_request)
        src_raster_subset = rasterio.open(aoi.raster_raw_input)
        src_np_subset = src_raster_subset.read(bands_request)
        dest_raster_subset = rasterio.open(aoi.raster_dest)
        assert src_np_subset.shape[0] == dest_raster_subset.count
        dest_np_subset = dest_raster_subset.read()
        assert np.all(src_np_subset == dest_np_subset)
        aoi.close_raster()

    def test_multiband_input_band_selection_from_letters(self):
        """Tests error when selecting bands from a multiband raster using letters, not integers"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = data[0]
        bands_request = ["R", "G"]
        with pytest.raises(ValueError):
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=bands_request)
            aoi.close_raster()

    def test_multiband_input_band_selection_too_many(self):
        """Tests error when selecting too many bands from a multiband raster"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = data[0]
        bands_request = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=bands_request)

    def test_singleband_input(self):
        """Tests reading a singleband raster as input with ${dataset.bands} pattern"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-singleband_ci.csv")
        bands = ['R', 'G', 'B']
        row = next(iter(data))
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=bands)
        assert aoi.raster_name.stem == aoi.aoi_id
        assert aoi.raster_bands_request == bands
        aoi.close_raster()

    def test_stac_input(self):
        """Tests singleband raster referenced by stac item as input"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-stac_ci.csv")
        bands = ['red', 'green', 'blue']
        row = next(iter(data))
        aoi = AOI(
            raster=row['tif'], label=row['gpkg'], split=row['split'],
            raster_bands_request=bands)
        assert aoi.raster_bands_request == bands
        for band, actual_raster in zip(["R", "G", "B"], aoi.raster_parsed):
            root_raster = "http://datacube-stage-data-public.s3.ca-central-1.amazonaws.com/store/imagery/optical/" \
                          "spacenet-samples/"
            exp_raster = root_raster + f'SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-{band}.tif'
            assert exp_raster == actual_raster
        aoi.close_raster()

    def test_stac_url_input(self):
        """Tests download of singleband raster as url path referenced by a stac item"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-singleband-url_ci.csv")
        row = next(iter(data))
        aoi = AOI(
            raster=row['tif'], label=row['gpkg'], split=row['split'],
            raster_bands_request=['R'], download_data=True, root_dir="data"
        )
        assert aoi.download_data is True
        assert Path("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-R.tif").is_file()
        aoi.close_raster()
        try:
            os.remove("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-R.tif")
        except PermissionError:
            pass

    def test_missing_label(self):
        """Tests error when provided label file is missing"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = next(iter(data))
        row['gpkg'] = "missing_file.gpkg"
        with pytest.raises(AttributeError):
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
            aoi.close_raster()

    def test_no_label(self):
        """Test when no label are provided. Should pass for inference. """
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        csv_path = "tests/inference/inference_segmentation_multiclass_no_label.csv"
        aois = aois_from_csv(csv_path=csv_path, bands_requested=[1, 2, 3])
        assert aois[0].label is None

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
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = next(iter(data))
        row['tif'] = "tests/data/massachusetts_buildings_kaggle/corrupt_file.tif"
        with pytest.raises(BaseException):
            aoi = AOI(raster=row['tif'], label=None)
            aoi.close_raster()

    def test_image_only(self) -> None:
        """Tests AOI creation with image only, ie no label"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = next(iter(data))
        aoi = AOI(raster=row['tif'], label=None)
        assert aoi.label is None
        aoi.close_raster()

    def test_filter_gdf_by_attribute(self):
        """Tests filtering features from a vector file according to an attribute field and value"""
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        data = read_csv("tests/tiling/tiling_segmentation_multiclass_ci.csv")
        iterator = iter(data)
        row = next(iterator)
        aoi = AOI(
            raster=row['tif'],
            label=row['gpkg'],
            split=row['split'],
            attr_field_filter="Quatreclasses",
            attr_values_filter=[4],  # buildings
        )
        assert np.array_equal(aoi.label_gdf_filtered, aoi.label_gdf[aoi.label_gdf.Quatreclasses == '4'])
        aoi.close_raster()

    def test_missing_raster(self) -> None:
        """Tests error when pointing to missing raster"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = next(iter(data))
        row['tif'] = "missing_raster.tif"
        with pytest.raises(RasterioIOError):
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
            aoi.close_raster()

    def test_wrong_split(self) -> None:
        """Tests error when setting a wrong split, ie not 'trn', 'tst' or 'inference'"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = next(iter(data))
        row['split'] = "missing_split"
        with pytest.raises(ValueError):
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
            aoi.close_raster()

    def test_download_data(self) -> None:
        """Tests download data"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = next(iter(data))
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
        data = read_csv("tests/tiling/tiling_segmentation_binary-stac_ci.csv")
        row = next(iter(data))
        with pytest.raises(ValueError):
            aoi = AOI(
                raster=row['tif'], label=row['gpkg'], split=row['split'],
                raster_bands_request=['ru', 'gris', 'but'])
            aoi.close_raster()

    def test_stac_input_empty_band_request(self):
        """Tests error when band selection is required (stac item) but missing"""
        extract_archive(src="tests/data/spacenet.zip")
        extract_archive(src="tests/data/massachusetts_buildings_kaggle.zip")
        raster_raw = (
            ("tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03.json", ""),
            ("tests/data/massachusetts_buildings_kaggle/22978945_15_uint8_clipped_${dataset.bands}.tif", ""),
        )
        for raster_raw, bands_requested in raster_raw:
            with pytest.raises((ValueError, TypeError)):
                AOI.parse_input_raster(csv_raster_str=raster_raw, raster_bands_requested=bands_requested)

    def test_no_intersection(self) -> None:
        """Tests error testing no intersection between raster and label"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = next(iter(data))
        row['gpkg'] = "tests/data/new_brunswick_aerial/BakerLake_2017_clipped.gpkg"
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
        assert aoi.bounds_iou == 0
        aoi.close_raster()

    def test_write_multiband_from_single_band(self) -> None:
        """Tests the 'write_multiband' method"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-singleband_ci.csv")
        row = data[0]
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['R', 'G', 'B'],
                  write_dest_raster=True, root_dir="data")
        assert Path("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-R-G-B.tif").is_file()
        aoi.close_raster()
        os.remove("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-R-G-B.tif")

    def test_write_multiband_from_single_band_url(self) -> None:
        """Tests the 'write_multiband' method with singleband raster as URL"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-singleband-url_ci.csv")
        row = next(iter(data))
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['R', 'G', 'B'],
                  write_dest_raster=True, root_dir="data", download_data=True)
        assert Path("data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-R-G-B.tif").is_file()
        assert aoi.download_data is True
        aoi.close_raster()
        for bands in ["R-G-B", "R", "G", "B"]:
            os.remove(f"data/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-{bands}.tif")

    def test_write_multiband_not_applicable(self) -> None:
        """Tests the skipping of 'write_multiband' method"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-multiband_ci.csv")
        row = next(iter(data))
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=[1, 2, 3],
                  write_dest_raster=True, root_dir="data")
        assert not aoi.raster_needs_vrt and not aoi.raster_is_vrt
        aoi.close_raster()

    def test_download_true_not_url(self) -> None:
        """Tests AOI creation if download_data set to True, but not necessary (local image)"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-singleband_ci.csv")
        row = next(iter(data))
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], download_data=True,
                  raster_bands_request=['R', 'G', 'B'])
        aoi.close_raster()

    def test_raster_stats_from_stac(self) -> None:
        """Tests the calculation of statistics of raster data as stac item from an AOI instance"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-stac_ci.csv")
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
        row = next(iter(data))
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=bands_request)
        stats = aoi.calc_raster_stats()
        for band, band_stat in stats.items():
            assert band_stat['statistics'] == expected_stats[band]['statistics']
            assert len(band_stat['histogram']['buckets']) == 256
        aoi.close_raster()

    def test_raster_stats_not_stac(self) -> None:
        """Tests the calculation of statistics of local multiband raster data from an AOI instance"""
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        data = read_csv("tests/tiling/tiling_segmentation_multiclass_ci.csv")
        expected_stats = {
            'band_0': {'statistics': {'minimum': 11, 'maximum': 254, 'mean': 159.36075617930456, 'median': 165.0,
                                      'std': 48.9924913616138}},
            'band_1': {'statistics': {'minimum': 12, 'maximum': 255, 'mean': 149.58768328445748, 'median': 154.0,
                                      'std': 46.204003828563714}},
            'band_2': {'statistics': {'minimum': 26, 'maximum': 254, 'mean': 119.60827398408044, 'median': 117.0,
                                      'std': 41.85516710316288}},
            'all': {'statistics': {'minimum': 16.333333333333332, 'maximum': 254.33333333333334,
                                   'mean': 142.8522378159475, 'median': 145.33333333333334, 'std': 45.68388743111347}}}
        row = next(iter(data))
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
        stats = aoi.calc_raster_stats()
        for band, band_stat in stats.items():
            assert band_stat['statistics'] == expected_stats[band]['statistics']
            assert len(band_stat['histogram']['buckets']) == 256
        aoi.close_raster()

    def test_to_dict(self):
        """Test the 'to_dict()' method on an AOI instance"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/tiling/tiling_segmentation_binary-stac_ci.csv")
        bands = ['red', 'green', 'blue']
        row = next(iter(data))
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=bands)
        aoi.to_dict()
        aoi.close_raster()

    def test_for_multiprocessing(self) -> None:
        """Tests multiprocessing on AOI instances"""
        extract_archive(src="tests/data/new_brunswick_aerial.zip")
        data = read_csv("tests/tiling/tiling_segmentation_multiclass_ci.csv")
        inputs = []
        row = next(iter(data))
        aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], for_multiprocessing=True)
        inputs.append([aoi_read_raster, aoi])
        assert aoi.raster_closed is True
        assert aoi.raster is None

        with multiprocessing.get_context('spawn').Pool(None) as pool:
            aoi_meta = pool.map_async(map_wrapper, inputs).get()
        print(aoi_meta)

    def test_name_raster(self) -> None:
        """Tests naming of raster given multiple input type"""
        inputs = [
            ("tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-${dataset.bands}.tif",
             ["R", "G", "B"]),
            ("http://datacube-stage-data-public.s3.ca-central-1.amazonaws.com/store/imagery/optical/spacenet-samples/"
             "SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-${dataset.bands}.tif", ["R", "G", "B"]),
            ("tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03.json", ["red", "green", "blue"]),
            ("tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB.tif", [1, 2, 3]),
            ("tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB.tif", [1, 2, 3, 4, 5, 6, 7]),
            ("tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB.tif", []),
        ]
        expected_list = [
            "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-R-G-B.tif",
            "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-R-G-B.tif",
            "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03_red-green-blue.tif",
            "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB_1-2-3.tif",
            "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB_7bands.tif",
            "tests/data/spacenet/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03-RGB.tif",
        ]
        for input, expected in zip(inputs, expected_list):
            actual = AOI.name_raster(root_dir="tests/data/spacenet/", input_path=input[0], bands_list=input[1])
            assert Path(expected) == actual


def map_wrapper(x):
    """For multi-threading"""
    return x[0](*(x[1:]))


def aoi_read_raster(aoi: AOI):
    """Function to package in multiprocessing"""
    aoi.raster = rasterio.open(aoi.raster_dest)
    return aoi.raster.meta
