import numpy as np
from pathlib import Path
import sys
from typing import Any, Dict, cast

from einops import rearrange
from osgeo import ogr
import rasterio
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.plot import reshape_as_image
from rasterio.windows import from_bounds
from rasterio.vrt import WarpedVRT
import torch
from torch.utils.data import Dataset
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox

from utils.logger import get_logger

# Set the logging file
logging = get_logger(__name__)  # import logging


def append_to_dataset(dataset, sample):
    """
    Append a new sample to a provided dataset. The dataset has to be expanded before we can add value to it.
    :param dataset:
    :param sample: data to append
    :return: Index of the newly added sample.
    """
    old_size = dataset.shape[0]  # this function always appends samples on the first axis
    dataset.resize(old_size + 1, axis=0)
    dataset[old_size, ...] = sample
    return old_size


class SegmentationDataset(Dataset):
    """Semantic segmentation dataset based on input csvs listing pairs of imagery and ground truth patches as .tif."""

    def __init__(self,
                 dataset_list_path,
                 dataset_type,
                 num_bands,
                 max_sample_count=None,
                 transforms=None,
                 augmentations=None,
                 debug=False):
        self.max_sample_count = max_sample_count
        self.dataset_type = dataset_type
        self.num_bands = num_bands
        self.transforms = transforms
        self.augmentations = augmentations
        self.debug = debug
        self.list_path = dataset_list_path
        if not Path(self.list_path).is_file():
            logging.error(f"Couldn't locate dataset list file: {self.list_path}.\n"
                          f"If purposely omitting test set, this error can be ignored")
            self.max_sample_count = 0
        else:
            with open(self.list_path, 'r') as datafile:
                datalist = datafile.readlines()
                if self.max_sample_count is None:
                    self.max_sample_count = len(datalist)

    def __len__(self):
        return self.max_sample_count

    def __getitem__(self, index):
        with open(self.list_path, 'r') as datafile:
            datalist = datafile.readlines()
            data_line = datalist[index]
            with rasterio.open(data_line.split(';')[0], 'r') as sat_handle:
                sat_img = reshape_as_image(sat_handle.read())
                metadata = sat_handle.meta
            with rasterio.open(data_line.split(';')[1].rstrip('\n'), 'r') as label_handle:
                map_img = reshape_as_image(label_handle.read())
                map_img = map_img[..., 0]

            assert self.num_bands <= sat_img.shape[-1]

        sample = {"image": sat_img, "mask": map_img, "metadata": metadata, "list_path": self.list_path}

        sample["image"] = rearrange(sample["image"], 'h w c -> c h w')

        # Comply to torchgeo's datasets standard: convert to torch tensor in Dataset's __getitem__
        # https://github.com/microsoft/torchgeo/blob/044d901dcae3badab1b22822180bab15e8dc2198/torchgeo/datasets/chesapeake.py#L640
        sample["image"] = torch.from_numpy(sample["image"])
        sample["mask"] = torch.from_numpy(sample["mask"]).float().unsqueeze(0)

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.augmentations is not None:
            sample = self.augmentations(sample)

        # torchmetrics expects masks to be longs without a channel dimension
        sample["mask"] = sample["mask"].squeeze(0).long()

        if self.debug:
            # assert no new class values in mask
            initial_class_ids = set(np.unique(map_img))
            final_class_ids = set(np.unique(sample['mask'].numpy()))
            if not final_class_ids.issubset(initial_class_ids):
                logging.warning(f"\nWARNING: Class values for label before and after augmentations don't match."
                                f"\nUnique values before: {initial_class_ids}"
                                f"\nUnique values after: {final_class_ids}"
                                f"\nIgnore if some augmentations have padded with dontcare value.")
        sample['index'] = index
        return sample


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

        tensor = torch.tensor(dest)

        return tensor


class GDLVectorDataset(GeoDataset):
    """The dataset is base on rasterio's DatasetReader class, instanciated by vector file."""
    def __init__(self, vec_ds: str = None, res: float = 0.0001) -> None:
        """Initialize a new Dataset instance.

        Args:
            vec_ds: vector labels geopackage
            res: resolution of the dataset in units of CRS
        Returns:
            An OGR datasource in memory
        """
        super().__init__()

        self.vec_ds = ogr.Open(str(vec_ds))
        self.res = res

        assert self.vec_ds is not None, "The vector dataset is empty."

        vec_ds_layer = self.vec_ds.GetLayer()
        minx, maxx, miny, maxy = vec_ds_layer.GetExtent()
        del vec_ds_layer

        mint = 0
        maxt = sys.maxsize
        coords = (minx, maxx, miny, maxy, mint, maxt)
        self.index.insert(0, coords, 'vec')

        src_mem_driver = ogr.GetDriverByName('MEMORY')
        self.mem_vec_ds = src_mem_driver.CopyDataSource(self.vec_ds, 'src_mem_ds')
        self.vec_srs = self.mem_vec_ds.GetLayer().GetSpatialRef()
        vec_srs_wkt = self.vec_srs.ExportToPrettyWkt()

        self._crs = CRS.from_wkt(vec_srs_wkt)

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
        ogr.Layer.Clip(self.mem_vec_ds.GetLayer(), mem_layer, out_layer)

        sample = {"mask": out_mem_ds, "crs": self.crs, "bbox": query}

        return sample
