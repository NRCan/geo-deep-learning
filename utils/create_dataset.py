import numpy as np
from pathlib import Path

import rasterio
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset

from utils.logger import get_logger

# These two import statements prevent exception when using eval(metadata) in SegmentationDataset()'s __init__()
from rasterio.crs import CRS
from affine import Affine

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
                 radiom_transform=None,
                 geom_transform=None,
                 totensor_transform=None,
                 debug=False):
        # note: if 'max_sample_count' is None, then it will be read from the dataset at runtime
        self.max_sample_count = max_sample_count
        self.dataset_type = dataset_type
        self.num_bands = num_bands
        self.radiom_transform = radiom_transform
        self.geom_transform = geom_transform
        self.totensor_transform = totensor_transform
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

            if isinstance(metadata, np.ndarray) and len(metadata) == 1:
                metadata = metadata[0]
            elif isinstance(metadata, bytes):
                metadata = metadata.decode('UTF-8')
            try:
                metadata = eval(metadata)
            except TypeError:
                pass

        sample = {"image": sat_img, "mask": map_img, "metadata": metadata, "list_path": self.list_path}

        if self.radiom_transform:  # radiometric transforms should always precede geometric ones
            sample = self.radiom_transform(sample)
        if self.geom_transform:  # rotation, geometric scaling, flip and crop. Will also put channels first and convert to torch tensor from numpy.
            sample = self.geom_transform(sample)

        sample = self.totensor_transform(sample)

        if self.debug:
            # assert no new class values in map_img
            initial_class_ids = set(np.unique(map_img))
            final_class_ids = set(np.unique(sample["mask"].numpy()))
            if not final_class_ids.issubset(initial_class_ids):
                logging.warning(f"\nWARNING: Class values for label before and after augmentations don't match."
                                f"\nUnique values before: {initial_class_ids}"
                                f"\nUnique values after: {final_class_ids}"
                                f"\nIgnore if some augmentations have padded with dontcare value.")
        sample['index'] = index
        return sample
