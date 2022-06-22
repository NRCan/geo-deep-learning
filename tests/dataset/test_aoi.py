import multiprocessing
from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
from shapely.geometry import box
from torchgeo.datasets.utils import extract_archive

from dataset.aoi import AOI
from utils.utils import read_csv


class Test_AOI(object):
    def test_multiband_input(self):
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])
            # TODO assert actual == expected

    def test_singleband_input(self):
        """Tests singleband input imagery with ${dataset.bands} pattern"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband_ci.csv")
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], raster_bands_request=['R', 'G', 'B'])
            # TODO assert actual == expected

    def test_stac_input(self):
        """Tests singleband input imagery from stac item"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-stac_ci.csv")
        for row in data:
            aoi = AOI(
                raster=row['tif'], label=row['gpkg'], split=row['split'],
                raster_bands_request=['red', 'green', 'blue'])
            # TODO assert actual == expected

    def test_stac_url_input(self):
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband-url_ci.csv")
        for row in data:
            aoi = AOI(
                raster=row['tif'], label=row['gpkg'], split=row['split'],
                raster_bands_request=['R'], download_data=True, root_dir="data"
            )
            # TODO assert actual == expected

    def test_missing_label(self):
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-multiband_ci.csv")
        for row in data:
            row['gpkg'] = "missing_file.gpkg"
            with pytest.raises(AttributeError):
                aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'])

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
            # TODO assert actual == expected

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

    def test_raster_stats_from_stac(self) -> None:
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
            break

    def test_raster_stats_not_stac(self) -> None:
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
            break
            
    def test_for_multiprocessing(self) -> None:
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_multiclass_ci.csv")
        inputs = []
        for row in data:
            aoi = AOI(raster=row['tif'], label=row['gpkg'], split=row['split'], for_multiprocessing=True)
            inputs.append([aoi_read_raster, aoi])

        with multiprocessing.get_context('spawn').Pool(None) as pool:
            aoi_meta = pool.map_async(map_wrapper, inputs).get()
        print(aoi_meta)


def map_wrapper(x):
    """For multi-threading"""
    return x[0](*(x[1:]))


def aoi_read_raster(aoi: AOI):
    aoi.raster_to_multiband()
    return aoi.raster.meta
