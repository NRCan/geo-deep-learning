import sys
from pathlib import Path
from typing import Any, Dict, List, cast

import kornia as K
import numpy as np
import pandas as pd
import rasterio
import torch
from affine import Affine
from osgeo import ogr
# These two import statements prevent exception when using eval(metadata) in SegmentationDataset()'s __init__()
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.plot import reshape_as_image
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from torch.utils.data import Dataset
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox

from utils.logger import get_logger

# Set the logging file
logging = get_logger(__name__)  # import logging


class SegmentationDataset(Dataset):
    """Semantic segmentation dataset based on input csvs listing pairs of imagery and ground truth patches as .tif.

    Args:
        dataset_list_path (str): The path to the dataset list file.
        num_bands (int): The number of bands in the imagery.
        dontcare (Optional[int]): The value to be ignored in the label.
        max_sample_count (Optional[int]): The maximum number of samples to load from the dataset.
        radiom_transform (Optional[Callable]): The radiometric transform function to be applied to the samples.
        geom_transform (Optional[Callable]): The geometric transform function to be applied to the samples.
        totensor_transform (Optional[Callable]): The transform function to convert samples to tensors.
        debug (bool): Whether to enable debug mode.

    Attributes:
        max_sample_count (int): The maximum number of samples to load from the dataset.
        num_bands (int): The number of bands in the imagery.
        radiom_transform (Optional[Callable]): The radiometric transform function to be applied to the samples.
        geom_transform (Optional[Callable]): The geometric transform function to be applied to the samples.
        totensor_transform (Optional[Callable]): The transform function to convert samples to tensors.
        debug (bool): Whether debug mode is enabled.
        dontcare (Optional[int]): The value to be ignored in the label.
        list_path (str): The path to the dataset list file.
        assets (List[Dict[str, str]]): The list of filepaths to images and labels.

    """

    def __init__(self,
                 dataset_list_path,
                 num_bands,
                 dontcare=None,
                 max_sample_count=None,
                 radiom_transform=None,
                 geom_transform=None,
                 totensor_transform=None,
                 debug=False):
        self.max_sample_count = max_sample_count
        self.num_bands = num_bands
        self.radiom_transform = radiom_transform
        self.geom_transform = geom_transform
        self.totensor_transform = totensor_transform
        self.debug = debug
        self.dontcare = dontcare
        self.list_path = dataset_list_path
        self.parent_folder = Path(self.list_path).parent
        
        if not Path(self.list_path).is_file():
            logging.error(f"Couldn't locate dataset list file: {self.list_path}.\n"
                          f"If purposely omitting test set, this error can be ignored")
        self.assets = self._load_data()
    
    def __len__(self):
        return len(self.assets)
    
    def __getitem__(self, index):
        
        sat_img, metadata = self._load_image(index)
        map_img = self._load_label(index)

        if isinstance(metadata, np.ndarray) and len(metadata) == 1:
            metadata = metadata[0]
        elif isinstance(metadata, bytes):
            metadata = metadata.decode('UTF-8')
        try:
            metadata = eval(metadata)
        except TypeError:
            pass
        
        sample = {"image": sat_img, "mask": map_img, "metadata": metadata, "list_path": self.list_path}
        # radiometric transforms should always precede geometric ones
        if self.radiom_transform:  
            sample = self.radiom_transform(sample)
        # rotation, geometric scaling, flip and crop. 
        # Will also put channels first and convert to torch tensor from numpy.
        if self.geom_transform:
            sample = self.geom_transform(sample)
        if self.totensor_transform:
            sample = self.totensor_transform(sample)

        if self.debug:
            # assert no new class values in map_img
            initial_class_ids = set(np.unique(map_img))
            if self.dontcare is not None:
                initial_class_ids.add(self.dontcare)
            final_class_ids = set(np.unique(sample['mask'].numpy()))
            if not final_class_ids.issubset(initial_class_ids):
                logging.debug(f"WARNING: Class ids for label before and after augmentations don't match. "
                              f"Ignore if overwritting ignore_index in ToTensorTarget")
                logging.warning(f"\nWARNING: Class values for label before and after augmentations don't match."
                                f"\nUnique values before: {initial_class_ids}"
                                f"\nUnique values after: {final_class_ids}"
                                f"\nIgnore if some augmentations have padded with dontcare value.")
        sample['index'] = index

        return sample
    
    def _load_data(self) -> List[str]:
        """Load the filepaths to images and labels
        
        Returns:
            List[str]: a list of filepaths to train/test data  
        """
        df = pd.read_csv(self.list_path, sep=';', header=None, usecols=[i for i in range(2)])
        assets = [{"image": x, "label": y} for x, y in zip(df[0], df[1])]
        
        return assets
    
    def _load_image(self, index: int):
        """ Load image 

        Args:
            index: poosition of image

        Returns:
            image array and metadata
        """
        image_path = self.parent_folder.joinpath(self.assets[index]["image"])
        with rasterio.open(image_path, 'r') as image_handle:
            image = reshape_as_image(image_handle.read())
            metadata = image_handle.meta
        assert self.num_bands <= image.shape[-1]
        
        return image, metadata
    
    def _load_label(self, index: int):
        """ Load label 

        Args:
            index: poosition of label

        Returns:
            label array and metadata
        """
        label_path = self.parent_folder.joinpath(self.assets[index]["label"])
        with rasterio.open(label_path, 'r') as label_handle:
                label = reshape_as_image(label_handle.read())
                label = label[..., 0]
        
        return label
    

class DRDataset(GeoDataset):
    def __init__(self, dr_ds: DatasetReader) -> None:
        """Initialize a new DRDataset instance.
        The dataset is base on rasterio's DatasetReader class, instanciated by rasterio.open().

        Args:
            dr_ds: DatasetReader object (rasterio)
        """
        super().__init__()

        self.dr_ds = dr_ds
        try:
            self.cmap = dr_ds.colormap(1)
        except ValueError:
            pass

        crs = dr_ds.crs
        res = dr_ds.res[0]

        with WarpedVRT(dr_ds, crs=crs) as dr:
            minx, miny, maxx, maxy = dr.bounds

        mint: float = 0
        maxt: float = sys.maxsize

        coords = (minx, maxx, miny, maxy, mint, maxt)
        self.index.insert(0, coords, 'dr')

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image and metadata at that index
        """
        data = self._get_tensor(query)
        key = "image"
        sample = {key: data, "crs": self.crs, "bbox": query}

        return sample

    def _get_tensor(self, query):
        """
        Get a patch based on the given query (bounding box).
        Args:
            query:

        Returns: Torch tensor patch.

        """
        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        out_width = round((query.maxx - query.minx) / self.res)
        out_height = round((query.maxy - query.miny) / self.res)
        out_shape = (self.dr_ds.count, out_height, out_width)

        dest = self.dr_ds.read(
            out_shape=out_shape, window=from_bounds(*bounds, self.dr_ds.transform)
        )

        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)

        return tensor


class GDLVectorDataset(GeoDataset):
    """The dataset is base on rasterio's DatasetReader class, instanciated by vector file."""
    def __init__(self, vec_ds: str = None, nodata_mask: ogr.DataSource | None = None, res: float = 0.0001) -> None:
        """Initialize a new Dataset instance.

        Args:
            vec_ds: vector labels geopackage
            nodata_mask: nodata mask taken from the corresponding raster image
            res: resolution of the dataset in units of CRS
        Returns:
            Initializes vector dataset (subclass of the GeoDataset).
        """
        super().__init__()

        # Read the dataset as an OGR datasouce:
        vec_ds = ogr.Open(str(vec_ds))
        self.res = res

        assert vec_ds is not None, "The vector dataset is empty."

        # Copy the datasource to memory:
        src_mem_driver = ogr.GetDriverByName('MEMORY')
        mem_vec_ds = src_mem_driver.CopyDataSource(vec_ds, 'src_mem_ds')
        self.vec_srs = mem_vec_ds.GetLayer().GetSpatialRef()
        vec_srs_wkt = self.vec_srs.ExportToPrettyWkt()
        self._crs = CRS.from_wkt(wkt=vec_srs_wkt)

        # Clip vector labels with raster nodata mask:
        self.ds = self._clip_nodata(vec_ds=mem_vec_ds, nodata_mask=nodata_mask)
        vec_ds_layer = self.ds.GetLayer()
        minx, maxx, miny, maxy = vec_ds_layer.GetExtent()
        del vec_ds_layer

        mint = 0
        maxt = sys.maxsize
        coords = (minx, maxx, miny, maxy, mint, maxt)
        self.index.insert(0, coords, 'vec')

        vec_ds = None
        mem_vec_ds = None

    def _clip_nodata(self, vec_ds: ogr.DataSource, nodata_mask: ogr.DataSource | None) -> ogr.DataSource:
        """
        Clips the vector dataset with the nodata mask polygon taken from the corresponding raster file.
        If the mask is None, returns non-clipped dataset.
        Args:
            vec_ds: OGR datasouce to clip
            nodata_mask: OGR vector noadata mask from the raster file, or None

        Returns:
            Clipped (or original) OGR datasource.
        """
        if nodata_mask is None:
            return vec_ds
        # Clip the dataset with the nodata mask:
        out_driver = ogr.GetDriverByName('MEMORY')
        cropped_mem_ds = out_driver.CreateDataSource('memdata')
        cropped_layer = cropped_mem_ds.CreateLayer('0', self.vec_srs, geom_type=ogr.wkbMultiPolygon)
        ogr.Layer.Clip(vec_ds.GetLayer(), nodata_mask.GetLayer(), cropped_layer)

        return cropped_mem_ds

    @staticmethod
    def _check_curve(layer: ogr.Layer) -> None:
        """
        This function validates that all features of the output patches are polygonal.
        Args:
            layer: OGR Layer object

        Returns:
            Replaces curve feature geometries with the approximated ones.

        """
        # Check if the feature geometry is polygonal:
        feature_defn = layer.GetLayerDefn()
        layer.ResetReading()
        feature = layer.GetNextFeature()
        while feature is not None:
            geom = feature.GetGeometryRef()
            name_wkt = geom.ExportToWkt()

            # Approximate a curvature by a polygon geometry:
            if 'curv' in name_wkt.lower():
                linear_geom = geom.GetLinearGeometry()
                new_feature = ogr.Feature(feature_defn)
                new_feature.SetGeometryDirectly(linear_geom)
                layer.CreateFeature(new_feature)
                layer.DeleteFeature(feature.GetFID())

            feature = layer.GetNextFeature()

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample as a OGR datasource in memory and metadata at that index
        """
        poly_box = ogr.Geometry(ogr.wkbLinearRing)
        poly_box.AddPoint(query.minx, query.maxy)
        poly_box.AddPoint(query.maxx, query.maxy)
        poly_box.AddPoint(query.maxx, query.miny)
        poly_box.AddPoint(query.minx, query.miny)
        poly_box.AddPoint(query.minx, query.maxy)
        # Create a Polygon object from the ring.
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(poly_box)

        # # Create a vector datasource in memory:
        mem_driver = ogr.GetDriverByName('MEMORY')
        mem_ds = mem_driver.CreateDataSource('memdata')
        mem_layer = mem_ds.CreateLayer('0', self.vec_srs, geom_type=ogr.wkbPolygon)
        feature_def = mem_layer.GetLayerDefn()
        out_feature = ogr.Feature(feature_def)
        # Set new geometry from the Polygon object (bounding box):
        out_feature.SetGeometry(poly)
        # Add new feature to output Layer
        mem_layer.CreateFeature(out_feature)

        # Crate the output vector patch datasource:
        out_driver = ogr.GetDriverByName('MEMORY')
        out_mem_ds = out_driver.CreateDataSource('memdata')
        # Clip it with the bounding box:
        out_layer = out_mem_ds.CreateLayer('0', self.vec_srs, geom_type=ogr.wkbMultiPolygon)
        ogr.Layer.Clip(self.ds.GetLayer(), mem_layer, out_layer)

        # Check that there is no curve geometry in the output patch:
        self._check_curve(layer=out_layer)

        sample = {"mask": out_mem_ds, "crs": self.crs, "bbox": query}

        return sample