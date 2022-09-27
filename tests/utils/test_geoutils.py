import numpy as np
import rasterio
from torchgeo.datasets.utils import extract_archive

from dataset.aoi import AOI
from utils.utils import read_csv


class TestGeoutils(object):
    def test_multiband_vrt_from_single_band(self) -> None:
        """Tests the 'stack_singlebands_vrt' utility"""
        extract_archive(src="tests/data/spacenet.zip")
        data = read_csv("tests/sampling/sampling_segmentation_binary-singleband_ci.csv")
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
