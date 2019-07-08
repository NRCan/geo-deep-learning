from shapely.geometry import shape
from owslib.wcs import WebCoverageService
import fiona
import math
import rasterio
from rasterio.merge import merge
import tempfile


def create_bbox_from_pol(vector_file):
    """
    Create a list of bounding boxes from a vector file.
    :param vector_file: (str) full file path of the vector file, containing AOIs.
    :return: (list) list of dict containing info on the dataset (trn or val), bbox and crs.
    """
    lst_bbox = []
    with fiona.open(vector_file, 'r') as src:
        for vector in src:
            bbox = shape(vector['geometry']).envelope
            lst_bbox.append({'bbox': bbox.bounds, 'crs': src.crs})
    return lst_bbox


def cut_bbox_from_maxsize(bbox, maxsize, resolution):
    """
    Split a bbox, based on the maxsize of the wcs service and the spatial resolution specified by the user.
    :param bbox: (tuple) Extent of the bbox.
    :param maxsize: (int) Maximum width and height (in pixel) that can be returned by the WCS service.
    :param resolution: (int) Spatial resolution requested by the user, to returned by the WCS service.
    :return: (list) list of bbox tuples
    """
    lst_sub_bbox = []
    xmin = math.floor(bbox[0])
    ymin = math.floor(bbox[1])
    xmax = math.ceil(bbox[2])
    ymax = math.ceil(bbox[3])

    chunk = math.floor(maxsize * resolution)

    for x in range(xmin, xmax, chunk):
        for y in range(ymin, ymax, chunk):
            sub_bbox = tuple((x, y, x+chunk, y+chunk))
            lst_sub_bbox.append(sub_bbox)
    return lst_sub_bbox


def merge_wcs_tiles(list_tiles):
    """
    Merge a list of wcs responses into one ndarray
    :param list_tiles: (list) list of temp tif full file paths.
    :return: (ndarray) mosaic.
    """
    src_files_to_mosaic = []
    for i in list_tiles:
        src = rasterio.open(i)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)
    return mosaic, out_trans


def wcs_request(bbox, service_url, version, coverage, epsg, resolution, output_format):
    """
    Perform a WCS get coverage request and save to temporary file the response from the server.
    :param bbox: (tuple) Bounding box for the request. (xmin, ymin, xmax, ymax)
    :param service_url: (str) URL of the WCS
    :param version: (str) Version of the service. Only supports version 1, for now.
    :param coverage: (str) Name of the layer to request.
    :param epsg: (str) EPSG code of the service.
    :param resolution: (int or str) Pixel size (x and y) for the request.
    :param output_format: (str) output format of the request.
    :return: (str) temp file name. 
    """
    url_wcs = WebCoverageService(service_url, version=version)

    if version.startswith('1.'):
        response_wcs = url_wcs.getCoverage(identifier=coverage, bbox=bbox, format=output_format, crs=f"urn:ogc:def:crs:EPSG::{epsg}", resx=resolution, resy=resolution)
    elif version.startswith('2.'):
        raise ValueError(f"Support for WCS 2.0.x has not been implemented yet. Please use WCS version 1.")

    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(response_wcs.read())
    return tmpfile.name
