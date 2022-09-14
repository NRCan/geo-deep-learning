import collections
import logging

import numpy as np
import pystac
import rasterio
from rasterio import MemoryFile
from rasterio.plot import reshape_as_raster
from rasterio.shutil import copy as riocopy
import xml.etree.ElementTree as ET

from solaris.utils.core import _check_rasterio_im_load

logger = logging.getLogger(__name__)


def create_new_raster_from_base(input_raster, output_raster, write_array):
    """Function to use info from input raster to create new one.
    Args:
        input_raster: input raster path and name
        output_raster: raster name and path to be created with info from input
        write_array (optional): array to write into the new raster

    Return:
        none
    """
    src = _check_rasterio_im_load(input_raster)
    if len(write_array.shape) == 2:  # 2D array
        count = 1
    elif len(write_array.shape) == 3:  # 3D array
        if write_array.shape[0] > 100:
            logging.warning(f"\nGot {write_array.shape[0]} bands. "
                            f"\nMake sure array follows rasterio's channels first convention")
            write_array = reshape_as_raster(write_array)
        count = write_array.shape[0]
    else:
        raise ValueError(f'Array with {len(write_array.shape)} dimensions cannot be written by rasterio.')

    # Cannot write to 'VRT' driver
    driver = 'GTiff' if src.driver == 'VRT' else src.driver

    with rasterio.open(output_raster, 'w',
                       driver=driver,
                       width=src.width,
                       height=src.height,
                       count=count,
                       crs=src.crs,
                       dtype=np.uint8,
                       transform=src.transform) as dst:
        if count == 1:
            dst.write(write_array[:, :], 1)
        else:
            dst.write(write_array)


def get_key_recursive(key, config):
    """Returns a value recursively given a dictionary key that may contain multiple subkeys."""
    if not isinstance(key, list):
        key = key.split("/")  # subdict indexing split using slash
    assert key[0] in config, f"missing key '{key[0]}' in metadata dictionary: {config}"
    val = config[key[0]]
    if isinstance(val, (dict, collections.OrderedDict)):
        assert len(key) > 1, "missing keys to index metadata subdictionaries"
        return get_key_recursive(key[1:], val)
    return int(val)


def is_stac_item(path: str) -> bool:
    """Checks if an input string or object is a valid stac item"""
    if isinstance(path, pystac.Item):
        return True
    else:
        try:
            pystac.Item.from_file(str(path))
            return True
        # with .tif as url, pystac/stac_io.py/read_test_from_href() returns Exception, not HTTPError
        except Exception:
            return False


def stack_singlebands_vrt(srcs, band=1):
    """
    Stacks multiple single-band raster into a single multiband virtual raster
    Source: https://gis.stackexchange.com/questions/392695/is-it-possible-to-build-a-vrt-file-from-multiple-files-with-rasterio
    @param srcs:
        List of paths/urls to single-band raster
    @param band:
        TODO
    @return:
        RasterDataset object containing VRT
    """
    vrt_bands = []
    for srcnum, src in enumerate(srcs, start=1):
        with rasterio.open(src) as ras, MemoryFile() as mem:
            riocopy(ras, mem.name, driver='VRT')
            vrt_xml = mem.read().decode('utf-8')
            vrt_dataset = ET.fromstring(vrt_xml)
            for bandnum, vrt_band in enumerate(vrt_dataset.iter('VRTRasterBand'), start=1):
                if bandnum == band:
                    vrt_band.set('band', str(srcnum))
                    vrt_bands.append(vrt_band)
                    vrt_dataset.remove(vrt_band)
    for vrt_band in vrt_bands:
        vrt_dataset.append(vrt_band)

    return ET.tostring(vrt_dataset).decode('UTF-8')
