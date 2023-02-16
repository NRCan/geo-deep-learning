import collections
import logging
from distutils.version import LooseVersion
from pathlib import Path
from typing import List, Union, Sequence

import pyproj
from fiona._err import CPLE_OpenFailedError
from fiona.errors import DriverError
import geopandas as gpd
import numpy as np
import pystac
import rasterio
from hydra.utils import to_absolute_path
from pandas.io.common import is_url
from rasterio import MemoryFile, DatasetReader
from rasterio.plot import reshape_as_raster
from rasterio.shutil import copy as riocopy
import xml.etree.ElementTree as ET
from osgeo import gdal, gdalconst, ogr, osr

from shapely.geometry import box, Polygon

logger = logging.getLogger(__name__)


def create_new_raster_from_base(input_raster, output_raster, write_array, dtype = np.uint8, **kwargs):
    """Function to use info from input raster to create new one.
    Args:
        input_raster: input raster path and name
        output_raster: raster name and path to be created with info from input
        write_array (optional): array to write into the new raster
        dtype (optional): data type of output raster
        kwargs (optional): Complementary parameter(s)

    Return:
        None
    """
    src = check_rasterio_im_load(input_raster)
    if len(write_array.shape) == 2:  # 2D array
        count = 1
        write_array = write_array[np.newaxis, :, :]
    elif len(write_array.shape) == 3:  # 3D array
        if write_array.shape[0] > 100:
            logging.warning(f"\nGot {write_array.shape[0]} bands. "
                            f"\nMake sure array follows rasterio's channels first convention")
            write_array = reshape_as_raster(write_array)
        count = write_array.shape[0]
    else:
        raise ValueError(f'Array with {len(write_array.shape)} dimensions cannot be written by rasterio.')

    if write_array.shape[-2:] != (src.height, src.width):
        raise ValueError(f"Output array's width and height should be identical to dimensions of input reference raster"
                         f"\nInput reference raster shape (h x w): ({src.height}, {src.width})"
                         f"\nOutput array shape (h x w): {write_array.shape[1:]}")
    # Cannot write to 'VRT' driver
    driver = 'GTiff' if src.driver == 'VRT' else src.driver

    with rasterio.open(output_raster, 'w',
                       driver=driver,
                       width=src.width,
                       height=src.height,
                       count=count,
                       crs=src.crs,
                       dtype=dtype,
                       transform=src.transform,
                       compress='lzw') as dst:
        dst.write(write_array)
        # add tag to transmit more informations
        if 'checkpoint_path' in kwargs.keys():
            # add the path to the model checkpoint
            dst.update_tags(checkpoint=kwargs['checkpoint_path'])


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


def stack_singlebands_vrt(srcs: List, band: int = 1):
    """
    Stacks multiple single-band raster into a single multiband virtual raster
    Source: https://gis.stackexchange.com/questions/392695/is-it-possible-to-build-a-vrt-file-from-multiple-files-with-rasterio
    @param srcs:
        List of paths/urls to single-band rasters
    @param band:
        Index of band from source raster to stack into multiband VRT (index starts at 1 per GDAL convention)
    @return:
        RasterDataset object containing VRT
    """
    vrt_bands = []
    for srcnum, src in enumerate(srcs, start=1):
        with check_rasterio_im_load(src) as ras, MemoryFile() as mem:
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


def subset_multiband_vrt(src: Union[str, Path], band_request: Sequence = []):
    """
    Creates a multiband virtual raster containing a subset of all available bands in a source multiband raster
    @param src:
        Path/url to a multiband raster
    @param band_request:
        Indices of bands from source raster to subset from source multiband (index starts at 1 per GDAL convention).
        Order matters, i.e. if source raster is BGR, "[3,2,1]" will create a VRT with bands as RGB
    @return:
        RasterDataset object containing VRT
    """
    if not isinstance(src, (str, Path)) and not Path(src).is_file():
        raise ValueError(f"Invalid source multiband raster.\n"
                         f"Got {src}")
    with rasterio.open(src) as ras, MemoryFile() as mem:
        riocopy(ras, mem.name, driver='VRT')
        vrt_xml = mem.read().decode('utf-8')
        vrt_dataset = ET.fromstring(vrt_xml)
        vrt_dataset_dict = {int(band.get('band')): band for band in vrt_dataset.iter("VRTRasterBand")}
        for band in vrt_dataset_dict.values():
            vrt_dataset.remove(band)

        for dest_band_idx, src_band_idx in enumerate(band_request, start=1):
            vrt_band = vrt_dataset_dict[src_band_idx]
            vrt_band.set('band', str(dest_band_idx))
            vrt_dataset.append(vrt_band)

    return ET.tostring(vrt_dataset).decode('UTF-8')


def check_rasterio_im_load(im):
    """
    Check if `im` is already loaded in; if not, load it in.
    Copied from: https://github.com/CosmiQ/solaris/blob/main/solaris/utils/core.py#L17
    """
    if isinstance(im, (str, Path)):
        if not is_url(im) and 'VRTDataset' not in str(im):
            im = to_absolute_path(str(im))
        return rasterio.open(im)
    elif isinstance(im, rasterio.DatasetReader):
        return im
    else:
        raise ValueError("{} is not an accepted image format for rasterio.".format(im))


def check_gdf_load(gdf):
    """
    Check if `gdf` is already loaded in, if not, load from geojson.
    Copied from: https://github.com/CosmiQ/solaris/blob/main/solaris/utils/core.py#L52
    """
    if isinstance(gdf, (str, Path)):
        if not is_url(gdf):
            gdf = to_absolute_path(str(gdf))
        # as of geopandas 0.6.2, using the OGR CSV driver requires some add'nal
        # kwargs to create a valid geodataframe with a geometry column. see
        # https://github.com/geopandas/geopandas/issues/1234
        if str(gdf).lower().endswith("csv"):
            return gpd.read_file(
                gdf, GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO"
            )
        try:
            return gpd.read_file(gdf)
        except (DriverError, CPLE_OpenFailedError):
            logging.warning(
                f"GeoDataFrame couldn't be loaded: either {gdf} isn't a valid"
                " path or it isn't a valid vector file. Returning an empty"
                " GeoDataFrame."
            )
            return gpd.GeoDataFrame()
    elif isinstance(gdf, gpd.GeoDataFrame):
        return gdf
    else:
        raise ValueError(f"{gdf} is not an accepted GeoDataFrame format.")


def check_crs(input_crs, return_rasterio=False):
    """Convert CRS to the ``pyproj.CRS`` object passed by ``solaris``."""
    if not isinstance(input_crs, pyproj.CRS) and input_crs is not None:
        out_crs = pyproj.CRS(input_crs)
    else:
        out_crs = input_crs

    if return_rasterio:
        if LooseVersion(rasterio.__gdal_version__) >= LooseVersion("3.0.0"):
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt())
        else:
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt("WKT1_GDAL"))

    return out_crs


def bounds_riodataset(raster: DatasetReader) -> box:
    """Returns bounds of a rasterio DatasetReader as shapely box instance"""
    return box(*list(raster.bounds))


def bounds_gdf(gdf: gpd.GeoDataFrame) -> box:
    """Returns bounds of a GeoDataFrame as shapely box instance"""
    if gdf.empty:
        return Polygon()
    gdf_bounds = gdf.total_bounds
    gdf_bounds_box = box(*gdf_bounds.tolist())
    return gdf_bounds_box


def overlap_poly1_rto_poly2(polygon1: Polygon, polygon2: Polygon) -> float:
    """Calculate intersection of extents from polygon 1 and 2 over extent of a polygon 2"""
    intersection = polygon1.intersection(polygon2).area
    return intersection / (polygon2.area + 1e-30)


def multi2poly(returned_vector_pred, layer_name=None):
    """
    Converts shapely multipolygon to polygon. If fails, returns a logging error.
    This function will read a PATH string, create a geodataframe, explode all
    multipolygon to polygon and save the geodataframe at the same PATH.
    Args:
        returned_vector_pred: string, geopackage PATH where the post-processing
                              results are saved.
        layer_name (optional): string, the name of layer to look into for multipolygons.
                               For example, if using during post-processing, the layer
                               name could represent the class name if class are stored
                               in separate layers. Default None.
                    
    Return:
        none
    """
    try: # Try to convert multipolygon to polygon
        df = gpd.read_file(returned_vector_pred, layer=layer_name)
        if 'MultiPolygon' in df['geometry'].geom_type.values:
            logging.info("\nConverting multiPolygon to Polygon...")
            gdf_exploded = df.explode(index_parts=True, ignore_index=True)
            gdf_exploded.to_file(returned_vector_pred, layer=layer_name) # overwrite the layer readed
    except Exception as e:
        logging.error(f"\nSomething went wrong during the conversion of Polygon. \nError {type(e)}: {e}")


def fetch_tag_raster(raster_path, tag_wanted):
    """
    Fetch the tag(s) information saved inside the tiff.
    TODO, change the `tag_wanted` to accept str or list of str.
    Args:
        raster_path: string, raster path
        tag_wanted: string, tag name, ex. 'checkpoint'
    Return:
        string containing associate information to the `tag_wanted`
    """
    with rasterio.open(raster_path) as tiff_src:
        tags = tiff_src.tags()
    # check if the tiff have the wanted tag save in
    if tag_wanted in tags.keys():
        return tags[tag_wanted]
    else:
        logging.error(
            f"\nThe tag {tag_wanted} was not found in the {tags.keys()},"
            f" try again with one inside that list."
        )
        raise ValueError('Tag not found.')


def gdf_mean_vertices_nb(gdf: gpd.GeoDataFrame):
    """
    Counts vertices of all polygons inside a given GeoDataFrame
    @param gdf: input GeoDataFrame to count vertices from
    """
    if len(gdf.geometry) == 0:
        print("No features in GeoDataFrame")
        return None
    vertices_per_polygon = []
    for geom in gdf.geometry:
        if geom is None:
            logging.warning(f"GeoDataFrame contains a \"None\" geometry")
        elif geom.geom_type == "MultiPolygon":
            for polygon in geom.geoms:
                vertices_per_polygon.append(len(polygon.exterior.coords))
        elif geom.geom_type == "Polygon":
            vertices_per_polygon.append(len(geom.exterior.coords))
        else:
            logging.warning(f"Only supports MultiPolygon or Polygon. \nGot {geom.geom_type}")
    mean_ext_vert_nb = np.mean(vertices_per_polygon)
    return mean_ext_vert_nb


def mask_nodata(img_patch: Union[str, Path], gt_patch: Union[str, Path], nodata_val: int, mask_val: int = 255) -> None:
    """
    Masks label raster file with "ignore_index" value where pixels are "nodata" in the corresponding raster image.
    Args:
        img_patch: raster tile image path
        gt_patch: raster tile label path
        nodata_val: nodata value
        mask_val: masking value (255 by default)

    Returns:
        Masks label tile or None if no nadata pixels
    """
    image_ds = gdal.Open(str(img_patch), gdalconst.GA_ReadOnly)
    image_arr = image_ds.ReadAsArray()
    nodata_mask = image_arr != nodata_val
    nodata_mask_flat = np.sum(nodata_mask, axis=0) != 0

    if nodata_mask_flat.min() == 1:
        image_ds = None
        return

    gt_patch_ds = gdal.Open(str(gt_patch), gdalconst.GA_Update)
    gt_patch_arr = gt_patch_ds.ReadAsArray()
    masked_gt_arr = np.where(nodata_mask_flat == 1, gt_patch_arr, mask_val)
    gt_patch_ds.GetRasterBand(1).WriteArray(masked_gt_arr)
    gt_patch_ds = None
    image_ds = None


def nodata_vec_mask(raster: rasterio.DatasetReader, nodata_val: int = None) -> ogr.DataSource | None:
    """
    Fetches nodata mask from the raster image.
    Args:
        raster: raster dataset (DatasetReader) object.
        nodata_val: either None or predefined by a user integer value.
        If None, tries to fetch nodata value from the raster.

    Returns:
        Either None or vector nodata mask as an OGR datasource.
    """
    if nodata_val is None:
        nodata_val = raster.nodata
        if not isinstance(nodata_val, int | float):
            return None

    # Get original CRS and transform:
    crs_wkt = raster.crs.to_wkt()
    crs_gt = raster.transform

    # Read the data and calculate a nodata mask:
    image_arr = raster.read()
    nodata_mask = image_arr != nodata_val
    nodata_mask_flat = np.sum(nodata_mask, axis=0) != 0
    nodata_mask_flat = nodata_mask_flat.astype('uint8')

    raster_drv = gdal.GetDriverByName("MEM")
    dst = raster_drv.Create("/vsimem/raster", int(nodata_mask_flat.shape[1]),
                            int(nodata_mask_flat.shape[0]), 1, gdal.GDT_Byte)

    gdal_src_gt = [crs_gt[2], crs_gt[0], crs_gt[1], crs_gt[5], crs_gt[3], crs_gt[4]]
    dst.SetGeoTransform(gdal_src_gt)
    dst.SetProjection(crs_wkt)
    dst.GetRasterBand(1).WriteArray(nodata_mask_flat)
    dst.GetRasterBand(1).SetNoDataValue(0)
    src_band = dst.GetRasterBand(1)

    # Create vector datasource in memory:
    drv = ogr.GetDriverByName("MEMORY")
    vec_ds = drv.CreateDataSource('memdata')

    # Initialize projection:
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(crs_wkt)
    layer = vec_ds.CreateLayer('0', spatial_ref, geom_type=ogr.wkbPolygon)

    # Vectorize the raster nodata mask:
    gdal.Polygonize(src_band, src_band, layer, -1, [], callback=None)

    return vec_ds


if __name__ == "__main__":
    pass
