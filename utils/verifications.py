def is_valid_geom(geom):
    """
    Checks to see if geometry is a valid GeoJSON geometry type or
    GeometryCollection.

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