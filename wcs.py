import argparse
from utils import read_parameters
from shapely.geometry import shape
import fiona
import math


def create_bbox_from_pol(vector_file, attribute_name):
    """
    Function to create a list of bounding boxes from a vector file.
    :param vector_file: (str) full file path of the vector file, containing AOIs.
    :param attribute_name: (str) name of the attribute in the vector file, containing info on the dataset.
    :return: (list) list of dict containing info on the dataset (trn or val), bbox and crs.
    """

    lst_bbox = []
    with fiona.open(vector_file, 'r') as src:
        for vector in src:
            bbox = shape(vector['geometry']).envelope
            lst_bbox.append({'dst': vector['properties'][attribute_name], 'bbox': bbox.bounds, 'crs': src.crs})
    return lst_bbox


def cut_bbox_from_maxsize(bbox, maxsize, resolution):
    """
    Function to split a bbox, based on the maxsize of the wcs service and the spatial resolution specified by the user.
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

    chunk = int(maxsize//resolution)

    for x in range(xmin, xmax + chunk, chunk):
        for y in range(ymin, ymax + chunk, chunk):
            sub_bbox = tuple((x, y, x+chunk, y+chunk))
            lst_sub_bbox.append(sub_bbox)
    return lst_sub_bbox


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

    for pol in lst_envelope:
        sub_bbox_lst = cut_bbox_from_maxsize(pol['bbox'], maxsize, res)

        # 3 faire les requêtes wcs.

        # 4 Fusionner les résultats des requetes.

        # 5 images_to_samples.py ou inference.py


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    params = read_parameters(args.ParamFile)
    main(params)


    # URL_WCS_CCCOT = 'https://dy0a51yzxbhxw.cloudfront.net/ows?service=wcs'
    # cccot_wcs = WebCoverageService(URL_WCS_CCCOT, version='1.0.0')
    # print(list(cccot_wcs.contents))
    # print(type(cccot_wcs))