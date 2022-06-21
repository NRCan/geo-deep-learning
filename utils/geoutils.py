import collections
import logging
from pathlib import Path

import numpy as np

import fiona
import os

import pystac
import rasterio
from rasterio import MemoryFile
from rasterio.features import is_valid_geom
from rasterio.shutil import copy as riocopy
import xml.etree.ElementTree as ET

from solaris.utils.core import _check_rasterio_im_load

logger = logging.getLogger(__name__)


def lst_ids(list_vector, attr_name, target_ids=None, merge_all=True):
    '''
    Generates a dictionary from a list of vectors where keys are class numbers and values are corresponding features in a list.
    :param list_vector: list of vectors as returned by fiona.open
    :param attr_name: Attribute containing the identifier for a vector (may contain slashes if recursive)
    :param target_ids: list of identifiers to burn from the vector file (None = use all)
    :param merge_all: defines whether all vectors should be burned with their identifiers in a
            single layer or in individual layers (in the order provided by 'target_ids')
    :return: list of tuples in format (vector, class_id).
    '''
    lst_vector_tuple = {}
    for vector in list_vector:
        try:
            att_val = int(get_key_recursive(attr_name, vector)) if attr_name is not None else None
        except KeyError:
            attr_name = "properties/" + attr_name
            att_val = int(get_key_recursive(attr_name, vector)) if attr_name is not None else None
        if target_ids is None or att_val in target_ids:
            if att_val not in lst_vector_tuple:
                lst_vector_tuple[att_val] = []
            if merge_all:
                # here, we assume that the id can be cast to int!
                lst_vector_tuple[att_val].append((vector['geometry'], int(att_val) if att_val is not None else 0))
            else:
                # if not merging layers, just use '1' as the value for each target
                lst_vector_tuple[att_val].append((vector['geometry'], 1))
    return lst_vector_tuple


def vector_to_raster(vector_file, input_image, out_shape, attribute_name, fill=0, attribute_values=None, merge_all=True):
    """Function to rasterize vector data.
    Args:
        vector_file: Path and name of reference GeoPackage
        input_image: Rasterio file handle holding the (already opened) input raster
        attribute_name: Attribute containing the identifier for a vector (may contain slashes if recursive)
        fill: default background value to use when filling non-contiguous regions
        attribute_values: list of identifiers to burn from the vector file (None = use all)
        merge_all: defines whether all vectors should be burned with their identifiers in a
            single layer or in individual layers (in the order provided by 'target_ids')

    Return:
        numpy array of the burned image
    """
    # Extract vector features to burn in the raster image
    # FIXME use geopandas
    with fiona.open(vector_file, 'r') as src:
        lst_vector = [vector for vector in src]

    # Sort feature in order to priorize the burning in the raster image (ex: vegetation before roads...)
    if attribute_name is not None:
        try:
            lst_vector.sort(key=lambda vector: get_key_recursive(attribute_name, vector))
        except KeyError:
            attribute_name = "properties/" + attribute_name
            lst_vector.sort(key=lambda vector: get_key_recursive(attribute_name, vector))

    lst_vector_tuple = lst_ids(list_vector=lst_vector, attr_name=attribute_name, target_ids=attribute_values,
                               merge_all=merge_all)

    if not lst_vector_tuple:
        raise ValueError("No vector features found")
    elif merge_all:
        np_label_raster = rasterio.features.rasterize([v for vecs in lst_vector_tuple.values() for v in vecs],
                                           fill=fill,
                                           out_shape=out_shape,
                                           transform=input_image.transform,
                                           dtype=np.int16)
    else:
        burned_rasters = [rasterio.features.rasterize(lst_vector_tuple[id],
                                                      fill=fill,
                                                      out_shape=out_shape,
                                                      transform=input_image.transform,
                                                      dtype=np.int16) for id in lst_vector_tuple]
        np_label_raster = np.stack(burned_rasters, axis=-1)

    # overwritte label values to make sure they are continuous
    if attribute_values:
        for index, target_id in enumerate(attribute_values):
            if index+1 == target_id:
                continue
            else:
                np_label_raster[np_label_raster == target_id] = (index + 1)

    return np_label_raster


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
    if not key[0] in config:
        raise KeyError(f"missing key '{key[0]}' in metadata dictionary: {config}")
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


def stack_vrts(srcs, band=1):
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
