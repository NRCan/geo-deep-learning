import collections
from pathlib import Path

import numpy as np

import fiona

import rasterio
from rasterio.features import is_valid_geom
from rasterio.mask import mask


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
        id = get_key_recursive(attr_name, vector) if attr_name is not None else None
        if target_ids is None or id in target_ids:
            if id not in lst_vector_tuple:
                lst_vector_tuple[id] = []
            if merge_all:
                # here, we assume that the id can be cast to int!
                lst_vector_tuple[id].append((vector['geometry'], int(id) if id is not None else 0))
            else:
                # if not merging layers, just use '1' as the value for each target
                lst_vector_tuple[id].append((vector['geometry'], 1))
    return lst_vector_tuple


def channels_redistribution(raster, src_order: tuple, dst_order: tuple):
    """ Reorganizes channels of given raster according to desired order
    raster: Rasterio file handle holding the (already opened) input raster
    src_order: tuple of ints where len(tuple) == num of channels
        source order of channels
    dst_order: tuple of ints where len(tuple) == num of channels
        destination order of channels
    """
    pass


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def clip_raster_with_gpkg(raster, gpkg, debug=False):
    """Clips input raster to limits of vector data in gpkg. Adapted from: https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
    raster: Rasterio file handle holding the (already opened) input raster
    gpkg: Path and name of reference GeoPackage
    debug: if True, output raster as given by this function is saved to disk
    """
    from shapely.geometry import box  # geopandas and shapely become a project dependency only during sample creation
    import geopandas as gpd
    # Get extent of gpkg data with fiona
    with fiona.open(gpkg, 'r') as src:
        gpkg_crs = src.crs
        assert gpkg_crs == raster.crs
        minx, miny, maxx, maxy = src.bounds  # ouest, nord, est, sud

    # Create a bounding box with Shapely
    bbox = box(minx, miny, maxx, maxy)

    # Insert the bbox into a GeoDataFrame
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0]) #, crs=gpkg_crs['init'])

    # Re-project into the same coordinate system as the raster data
    # geo = geo.to_crs(crs=raster.crs.data)

    # Get the geometry coordinates by using the function.
    coords = getFeatures(geo)

    # clip the raster with the polygon
    try:
        out_img, out_transform = mask(dataset=raster, shapes=coords, crop=True)
    except ValueError as e:  # if gpkg's extent outside raster: "ValueError: Input shapes do not overlap raster."
        # TODO: warning or exception? if warning, except must be set in images_to_samples
        raise

    out_meta = raster.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform})
    out_tif = Path(raster.name).parent / f"{Path(raster.name).stem}_clipped{Path(raster.name).suffix}"
    dest = rasterio.open(out_tif, "w", **out_meta)
    if debug:
        print(f"DEBUG: writing clipped raster to {out_tif}")
        dest.write(out_img)

    return out_img, dest


def vector_to_raster(vector_file, input_image, out_shape, attribute_name, fill=0, target_ids=None, merge_all=True):
    """Function to rasterize vector data.
    Args:
        vector_file: Path and name of reference GeoPackage
        input_image: Rasterio file handle holding the (already opened) input raster
        attribute_name: Attribute containing the identifier for a vector (may contain slashes if recursive)
        fill: default background value to use when filling non-contiguous regions
        target_ids: list of identifiers to burn from the vector file (None = use all)
        merge_all: defines whether all vectors should be burned with their identifiers in a
            single layer or in individual layers (in the order provided by 'target_ids')

    Return:
        numpy array of the burned image
    """
    # Extract vector features to burn in the raster image
    with fiona.open(vector_file, 'r') as src:
        lst_vector = [vector for vector in src]

    # Sort feature in order to priorize the burning in the raster image (ex: vegetation before roads...)
    if attribute_name is not None:
        lst_vector.sort(key=lambda vector: get_key_recursive(attribute_name, vector))

    lst_vector_tuple = lst_ids(list_vector=lst_vector, attr_name=attribute_name, target_ids=target_ids,
                               merge_all=merge_all)

    if merge_all:
        return rasterio.features.rasterize([v for vecs in lst_vector_tuple.values() for v in vecs],
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
        return np.stack(burned_rasters, axis=-1)


def create_new_raster_from_base(input_raster, output_raster, write_array):
    """Function to use info from input raster to create new one.
    Args:
        input_raster: input raster path and name
        output_raster: raster name and path to be created with info from input
        write_array (optional): array to write into the new raster

    Return:
        none
    """
    if len(write_array.shape) == 2:  # 2D array
        count = 1
    elif len(write_array.shape) == 3:  # 3D array  # FIXME: why not keep all bands?
        count = 3
    else:
        raise ValueError(f'Array with {len(write_array.shape)} dimensions cannot be written by rasterio.')

    with rasterio.open(input_raster, 'r') as src:
        with rasterio.open(output_raster, 'w',
                           driver=src.driver,
                           width=src.width,
                           height=src.height,
                           count=count,
                           crs=src.crs,
                           dtype=np.uint8,
                           transform=src.transform) as dst:
            if count == 1:
                dst.write(write_array[:, :], 1)
            elif count == 3:
                dst.write(write_array[:, :, :3])  # Take only first three bands assuming they are RGB.


def get_key_recursive(key, config):
    """Returns a value recursively given a dictionary key that may contain multiple subkeys."""
    if not isinstance(key, list):
        key = key.split("/")  # subdict indexing split using slash
    assert key[0] in config, f"missing key '{key[0]}' in metadata dictionary: {config}"
    val = config[key[0]]
    if isinstance(val, (dict, collections.OrderedDict)):
        assert len(key) > 1, "missing keys to index metadata subdictionaries"
        return get_key_recursive(key[1:], val)
    return val