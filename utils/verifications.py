import fiona
from tqdm import tqdm

from utils.utils import get_key_recursive


def is_valid_geom(geom):
    """
    Checks to see if geometry is a valid GeoJSON geometry type or
    GeometryCollection. For example, helps detect features with area of 0.

    Geometries must be non-empty, and have at least x, y coordinates.

    Note: only the first coordinate is checked for validity.

    Parameters
    ----------
    geom: an object that implements the geo interface or GeoJSON-like object

    Returns
    -------
    bool: True if object is a valid GeoJSON geometry type
    """

    geom_types = {'Point', 'MultiPoint', 'LineString', 'LinearRing',
                  'MultiLineString', 'Polygon', 'MultiPolygon'}

    if not geom: #Custom if statement.
        return False

    if 'type' not in geom:
        return False

    try:
        geom_type = geom['type']
        if geom_type not in geom_types.union({'GeometryCollection'}):
            return False

    except TypeError:
        return False

    if geom_type in geom_types:
        if 'coordinates' not in geom:
            return False

        coords = geom['coordinates']

        if geom_type == 'Point':
            # Points must have at least x, y
            return len(coords) >= 2

        if geom_type == 'MultiPoint':
            # Multi points must have at least one point with at least x, y
            return len(coords) > 0 and len(coords[0]) >= 2

        if geom_type == 'LineString':
            # Lines must have at least 2 coordinates and at least x, y for
            # a coordinate
            return len(coords) >= 2 and len(coords[0]) >= 2

        if geom_type == 'LinearRing':
            # Rings must have at least 4 coordinates and at least x, y for
            # a coordinate
            return len(coords) >= 4 and len(coords[0]) >= 2

        if geom_type == 'MultiLineString':
            # Multi lines must have at least one LineString
            return (len(coords) > 0 and len(coords[0]) >= 2 and
                    len(coords[0][0]) >= 2)

        if geom_type == 'Polygon':
            # Polygons must have at least 1 ring, with at least 4 coordinates,
            # with at least x, y for a coordinate
            return (len(coords) > 0 and len(coords[0]) >= 4 and
                    len(coords[0][0]) >= 2)

        if geom_type == 'MultiPolygon':
            # Muti polygons must have at least one Polygon
            return (len(coords) > 0 and len(coords[0]) > 0 and
                    len(coords[0][0]) >= 4 and len(coords[0][0][0]) >= 2)

    if geom_type == 'GeometryCollection':
        if 'geometries' not in geom:
            return False

        if not len(geom['geometries']) > 0:
            # While technically valid according to GeoJSON spec, an empty
            # GeometryCollection will cause issues if used in rasterio
            return False

        for g in geom['geometries']:
            if not is_valid_geom(g):
                return False  # short-circuit and fail early

    return True


def validate_num_classes(vector_file, num_classes, attribute_name, ignore_index, debug=True):    # used only in images_to_samples.py
    """Validate that the number of classes in the vector file corresponds to the expected number
    Args:
        vector_file: full file path of the vector image
        num_classes: number of classes set in config.yaml
        attribute_name: name of the value field representing the required classes in the vector image file

        value_field: name of the value field representing the required classes in the vector image file
        ignore_index: (int) target value that is ignored during training and does not contribute to the input gradient
    Return:
        None
    """

    distinct_att = set()
    with fiona.open(vector_file, 'r') as src:
        for feature in tqdm(src, leave=False, position=1, desc=f'Scanning features'):
            distinct_att.add(get_key_recursive(attribute_name, feature))  # Use property of set to store unique values

    detected_classes = len(distinct_att) - len([ignore_index]) if ignore_index in distinct_att else len(distinct_att)

    if detected_classes != num_classes:
        raise ValueError('The number of classes in the yaml.config {} is different than the number of classes in '
                         'the file {} {}'.format (num_classes, vector_file, str(list(distinct_att))))