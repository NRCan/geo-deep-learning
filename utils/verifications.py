from pathlib import Path
from typing import Union

import fiona
import rasterio
from rasterio.features import is_valid_geom
from tqdm import tqdm

from utils.CreateDataset import MetaSegmentationDataset
from utils.geoutils import lst_ids, get_key_recursive


def validate_num_classes(vector_file: Union[str, Path], num_classes: int, attribute_name: str, ignore_index: int):
    """Check that `num_classes` is equal to number of classes detected in the specified attribute for each GeoPackage.
    FIXME: this validation **will not succeed** if a Geopackage contains only a subset of `num_classes` (e.g. 3 of 4).
    Args:
        vector_file: full file path of the vector image
        num_classes: number of classes set in config_template.yaml
        attribute_name: name of the value field representing the required classes in the vector image file
        ignore_index: (int) target value that is ignored during training and does not contribute to the input gradient
    Return:
        List of unique attribute values found in gpkg vector file
    """

    distinct_att = set()
    with fiona.open(vector_file, 'r') as src:
        for feature in tqdm(src, leave=False, position=1, desc=f'Scanning features'):
            distinct_att.add(get_key_recursive(attribute_name, feature))  # Use property of set to store unique values

    detected_classes = len(distinct_att) - len([ignore_index]) if ignore_index in distinct_att else len(distinct_att)

    if detected_classes != num_classes:
        raise ValueError('The number of classes in the yaml.config {} is different than the number of classes in '
                         'the file {} {}'.format(num_classes, vector_file, str(list(distinct_att))))

    return distinct_att


def add_background_to_num_class(task: str, num_classes: int):  # FIXME temporary patch for num_classes problem.
    """
    Adds one to number of classes for all segmentation tasks.

    param task: (str) task to perform. Either segmentation or classification
    param num_classes: (int) number of classes in task

    Returns number of classes corrected (+1) if task is segmentation
    """
    if task == 'segmentation':
        # assume background is implicitly needed (makes no sense to predict with one class, for example.)
        # this will trigger some warnings elsewhere, but should succeed nonetheless
        return num_classes + 1  # + 1 for background
    elif task == 'classification':
        return num_classes
    else:
        raise NotImplementedError(f'Task should be either classification or segmentation. Got "{task}"')


def assert_num_bands(raster_path: Union[str, Path], num_bands: int, meta_map):
    """
    Assert number of bands found in raster is equal to desired number of bands
    :param raster_path: (str or Path) path to raster file
    :param num_bands: number of bands raster file is expected to have
    :param meta_map:
    """

    # FIXME: think this through. User will have to calculate the total number of bands including meta layers and
    #  specify it in yaml. Is this the best approach? What if metalayers are added on the fly ?
    with rasterio.open(raster_path, 'r') as raster:
        input_band_count = raster.meta['count'] + MetaSegmentationDataset.get_meta_layer_count(meta_map)

    assert input_band_count == num_bands, f"The number of bands in the input image ({input_band_count}) " \
                                          f"and the parameter 'number_of_bands' in the yaml file ({num_bands}) " \
                                          f"should be identical"


def assert_crs_match(raster_path: Union[str, Path], gpkg_path: Union[str, Path]):
    """
    Assert Coordinate reference system between raster and gpkg match.
    :param raster_path: (str or Path) path to raster file
    :param gpkg_path: (str or Path) path to gpkg file
    """
    with fiona.open(gpkg_path, 'r') as src:
        gpkg_crs = src.crs

    with rasterio.open(raster_path, 'r') as raster:
        raster_crs = raster.crs

    assert gpkg_crs == raster_crs, f"CRS mismatch: \n" \
                                   f"TIF file \"{raster_path}\" has {raster_crs} CRS; \n" \
                                   f"GPKG file \"{gpkg_path}\" has {src.crs} CRS."


def validate_features_from_gpkg(gpkg: Union[str, Path], attribute_name: str):
    """
    Validate features in gpkg file
    :param gpkg: (str or Path) path to gpkg file
    :param attribute_name: name of the value field representing the required classes in the vector image file
    """
    # TODO: test this with invalid features.
    invalid_features_list = []
    # Validate vector features to burn in the raster image
    with fiona.open(gpkg, 'r') as src:  # TODO: refactor as independent function
        lst_vector = [vector for vector in src]
    shapes = lst_ids(list_vector=lst_vector, attr_name=attribute_name)
    for index, item in enumerate(tqdm([v for vecs in shapes.values() for v in vecs], leave=False, position=1)):
        # geom must be a valid GeoJSON geometry type and non-empty
        geom, value = item
        geom = getattr(geom, '__geo_interface__', None) or geom
        if not is_valid_geom(geom):
            if lst_vector[index]["id"] not in invalid_features_list:  # ignore if feature is already appended
                invalid_features_list.append(lst_vector[index]["id"])
    return invalid_features_list
