import argparse
from utils import read_parameters
from shapely.geometry import shape
from owslib.wcs import WebCoverageService
import fiona
import math
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
import tempfile


def create_bbox_from_pol(vector_file, attribute_name=None):
    """
    Create a list of bounding boxes from a vector file.
    :param vector_file: (str) full file path of the vector file, containing AOIs.
    :param attribute_name: (str) name of the attribute in the vector file, containing info on the dataset.
    :return: (list) list of dict containing info on the dataset (trn or val), bbox and crs.
    """

    lst_bbox = []
    with fiona.open(vector_file, 'r') as src:
        for vector in src:
            bbox = shape(vector['geometry']).envelope
            if attribute_name:
                lst_bbox.append({'dst': vector['properties'][attribute_name], 'bbox': bbox.bounds, 'crs': src.crs})
            else:
                lst_bbox.append({'dst': 'tst', 'bbox': bbox.bounds, 'crs': src.crs})
    return lst_bbox


def cut_bbox_from_maxsize(bbox, maxsize, resx, resy):
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

    chunkx = int(maxsize // resx)
    chunky = int(maxsize // resy)

    for x in range(xmin, xmax + chunkx, chunkx):
        for y in range(ymin, ymax + chunky, chunky):
            sub_bbox = tuple((x, y, x+chunkx, y+chunky))
            lst_sub_bbox.append(sub_bbox)
    return lst_sub_bbox


def merge_wcs_tiles(list_tiles, crs):
    """
    Merge a list of wcs responses into one ndarray
    :param list_tiles: (list) list of temp tif full file paths.
    :param crs: (int) EPSG code of the requests.
    :return: (ndarray) mosaic.
    """
    src_files_to_mosaic = []
    for i in list_tiles:
        src = rasterio.open(i)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)
    return mosaic, out_trans

    # with rasterio.open('D:\Travail\GDL_et_WCS\mosaic.tif', 'w', driver='GTiff',
    #                    width=mosaic.shape[2],
    #                    height=mosaic.shape[2]
    #
    # ,
    #                    count=1,
    #                    crs=crs,
    #                    dtype=np.float64,
    #                    transform=out_trans) as rasterio_array:
    #         rasterio_array.write(mosaic)


def wcs_request(sub_bbox, service_url, version, coverage, epsg, resx, resy, output_format):
    url_wcs = WebCoverageService(service_url, version=version)
    response_wcs = url_wcs.getCoverage(identifier=coverage, bbox=sub_bbox, format=output_format,
                                       crs=f"urn:ogc:def:crs:EPSG::{epsg}", resx=resx, resy=resy)

    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(response_wcs.read())
    return tmpfile.name


def main(yaml_params):
    service_url = yaml_params['wcs_parameters']['service_url']
    version = yaml_params['wcs_parameters']['version']
    layer_name = yaml_params['wcs_parameters']['layer_name']
    crs = yaml_params['wcs_parameters']['crs']
    gpkg_file = yaml_params['wcs_parameters']['aoi_file']
    attr_name = yaml_params['wcs_parameters']['attribute']
    res = yaml_params['wcs_parameters']['res']
    maxsize = yaml_params['wcs_parameters']['maxsize']
    output_format = yaml_params['wcs_parameters']['output_format']

    lst_envelope = create_bbox_from_pol(gpkg_file, attr_name)

    # for pol in lst_envelope:
    #     sub_bbox_lst = cut_bbox_from_maxsize(pol['bbox'], maxsize, res)
    #
    #     list_response_array = []
    #     for sub_bbox in sub_bbox_lst:
    #         # faire les requetes WCS.
    #         wcs_response = wcs_request(sub_bbox, service_url, version, layer_name, crs, res, output_format)
    #         # list_response_array.append({'array': wcs_response, 'bbox': sub_bbox})
    #         list_response_array.append(wcs_response)

            # wcs_response = wcs_request(sub_bbox, service_url, version, layer_name, crs, res, output_format)
    wgs_bboxes = [(-66.556, 45.909, -66.2125, 46.2125),
                  (-66.556, 46.2125, -66.2125, 46.516),
                  (-66.2125, 45.909, -65.869, 46.2125),
                  (-66.2125, 46.2125, -65.869, 46.516)]
    # utm_bboxes = [(1151226.82, 5124632.16, 1173938.10, 5152424.70),
    #               (1151226.82, 5152424.70, 1173938.10, 5189023.83),
    #               (1173938.10, 5124632.16, 1208090.42, 5152424.70),
    #               (1173938.10, 5152424.70, 1208090.42, 5189023.83)]
    list_response_array = []
    for sub_bbox in wgs_bboxes:
        bboxes = wgs_bboxes
        wcs_response = wcs_request(sub_bbox, service_url, version, layer_name, crs, res, output_format, bboxes.index(sub_bbox))
        list_response_array.append(wcs_response)

    merged_wcs_array = merge_wcs_tiles(list_response_array, crs)

    # 4 Fusionner les résultats des requetes.

    # 5 images_to_samples.py ou inference.py


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.ParamFile)
    main(params)

    URL_WCS_CCCOT = 'https://dy0a51yzxbhxw.cloudfront.net/ows?service=wcs'
    cccot_wcs = WebCoverageService(URL_WCS_CCCOT, version='1.0.0')
    print(list(cccot_wcs.contents))
    print(type(cccot_wcs))
