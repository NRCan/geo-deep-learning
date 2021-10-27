import collections
import logging
import os
import warnings
from pathlib import Path
from typing import List, Union

import h5py
import rasterio
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset
import numpy as np

import models.coordconv
from utils.utils import get_key_def, ordereddict_eval, compare_config_yamls
from utils.geoutils import get_key_recursive

# These two import statements prevent exception when using eval(metadata) in SegmentationDataset()'s __init__()
from rasterio.crs import CRS
from affine import Affine

logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """Semantic segmentation dataset based on HDF5 parsing."""

    def __init__(self,
                 dataset_list_path,
                 dataset_type,
                 num_bands,
                 max_sample_count=None,
                 dontcare=None,
                 radiom_transform=None,
                 geom_transform=None,
                 totensor_transform=None,
                 params=None,
                 debug=False):
        # note: if 'max_sample_count' is None, then it will be read from the dataset at runtime
        self.max_sample_count = max_sample_count
        self.dataset_type = dataset_type
        self.num_bands = num_bands
        self.radiom_transform = radiom_transform
        self.geom_transform = geom_transform
        self.totensor_transform = totensor_transform
        self.debug = debug
        self.dontcare = dontcare
        self.list_path = dataset_list_path
        with open(self.list_path, 'r') as datafile:
            datalist = datafile.readlines()
            if self.max_sample_count is None:
                self.max_sample_count = len(datalist)

    def __len__(self):
        return self.max_sample_count

    def _remap_labels(self, map_img):
        # note: will do nothing if 'dontcare' is not set in constructor, or set to non-zero value # TODO: seems like a temporary patch... dontcare should never be == 0, right ?
        if self.dontcare is None or self.dontcare != 0:
            return map_img
        # for now, the current implementation only handles the original 'dontcare' value as zero
        # to keep the impl simple, we just reduce all indices by one so that 'dontcare' becomes -1
        assert map_img.dtype == np.int8 or map_img.dtype == np.int16 or map_img.dtype == np.int32
        map_img -= 1
        return map_img

    def __getitem__(self, index):
        with open(self.list_path, 'r') as datafile:
            datalist = datafile.readlines()
            data_line = datalist[index]
            with rasterio.open(data_line.split(';')[0], 'r') as sat_handle:
                sat_img = reshape_as_image(sat_handle.read())
                metadata = sat_handle.meta
            with rasterio.open(data_line.split(';')[1], 'r') as label_handle:
                map_img = reshape_as_image(label_handle.read())
                map_img = map_img[...,0]

            assert self.num_bands <= sat_img.shape[-1]
            map_img = self._remap_labels(map_img)

        sample = {"sat_img": sat_img, "map_img": map_img, "metadata": metadata, "list_path": self.list_path}

        if self.radiom_transform:  # radiometric transforms should always precede geometric ones
            sample = self.radiom_transform(sample)
        if self.geom_transform:  # rotation, geometric scaling, flip and crop. Will also put channels first and convert to torch tensor from numpy.
            sample = self.geom_transform(sample)

        sample = self.totensor_transform(sample)

        if self.debug:
            # assert no new class values in map_img
            initial_class_ids = set(np.unique(map_img))
            if self.dontcare is not None:
                initial_class_ids.add(self.dontcare)
            final_class_ids = set(np.unique(sample['map_img'].numpy()))
            if not final_class_ids.issubset(initial_class_ids):
                logging.debug(f"WARNING: Class ids for label before and after augmentations don't match. "
                              f"Ignore if overwritting ignore_index in ToTensorTarget")
        sample['index'] = index
        return sample