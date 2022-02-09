import os
import h5py
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset

from utils.utils import ordereddict_eval

# These two import statements prevent exception when using eval(metadata) in SegmentationDataset()'s __init__()
from rasterio.crs import CRS
from affine import Affine

# Set the logging file
from utils import utils
logging = utils.get_logger(__name__)  # import logging


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


def create_files_and_datasets(samples_size: int, number_of_bands: int, meta_map, samples_folder: Path, cfg: DictConfig):
    """
    Function to create the hdfs files (trn, val and tst).
    :param samples_size: size of individual hdf5 samples to be created
    :param number_of_bands: number of bands in imagery
    :param meta_map:
    :param samples_folder: (str) Path to the output folder.
    :param cfg: (dict) Parameters found in the yaml config file.
    :return: (hdf5 datasets) trn, val ant tst datasets.
    """
    real_num_bands = number_of_bands
    assert real_num_bands > 0, "invalid number of bands when accounting for meta layers"
    hdf5_files = []
    for subset in ["trn", "val", "tst"]:
        hdf5_file = h5py.File(os.path.join(samples_folder, f"{subset}_samples.hdf5"), "w")
        hdf5_file.create_dataset("sat_img", (0, samples_size, samples_size, real_num_bands), np.uint16,
                                 maxshape=(None, samples_size, samples_size, real_num_bands))
        hdf5_file.create_dataset("map_img", (0, samples_size, samples_size), np.int16,
                                 maxshape=(None, samples_size, samples_size))
        hdf5_file.create_dataset("meta_idx", (0, 1), dtype=np.int16, maxshape=(None, 1))
        try:
            hdf5_file.create_dataset("metadata", (0, 1), dtype=h5py.string_dtype(), maxshape=(None, 1))
            hdf5_file.create_dataset("sample_metadata", (0, 1), dtype=h5py.string_dtype(), maxshape=(None, 1))
            hdf5_file.create_dataset("params", (0, 1), dtype=h5py.string_dtype(), maxshape=(None, 1))
            append_to_dataset(hdf5_file["params"], repr(OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))))
        except AttributeError:
            logging.exception(f'Update h5py to version 2.10 or higher')
            raise
        hdf5_files.append(hdf5_file)
    return hdf5_files


class SegmentationDataset(Dataset):
    """Semantic segmentation dataset based on HDF5 parsing."""

    def __init__(self, work_folder,
                 dataset_type,
                 num_bands,
                 max_sample_count=None,
                 radiom_transform=None,
                 geom_transform=None,
                 totensor_transform=None,
                 debug=False):
        # note: if 'max_sample_count' is None, then it will be read from the dataset at runtime
        self.work_folder = work_folder
        self.max_sample_count = max_sample_count
        self.dataset_type = dataset_type
        self.num_bands = num_bands
        self.metadata = []
        self.radiom_transform = radiom_transform
        self.geom_transform = geom_transform
        self.totensor_transform = totensor_transform
        self.debug = debug
        self.hdf5_path = os.path.join(self.work_folder, self.dataset_type + "_samples.hdf5")
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            for i in range(hdf5_file["metadata"].shape[0]):
                metadata = hdf5_file["metadata"][i, ...]
                if isinstance(metadata, np.ndarray) and len(metadata) == 1:
                    metadata = metadata[0]
                    metadata = ordereddict_eval(metadata)
                self.metadata.append(metadata)
            if self.max_sample_count is None:
                self.max_sample_count = hdf5_file["sat_img"].shape[0]

            # load yaml used to generate samples
            hdf5_params = hdf5_file['params'][0, 0]
            hdf5_params = ordereddict_eval(hdf5_params)

    def __len__(self):
        return self.max_sample_count

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            sat_img = np.float32(hdf5_file["sat_img"][index, ...])
            assert self.num_bands <= sat_img.shape[-1]
            map_img = hdf5_file["map_img"][index, ...]
            meta_idx = int(hdf5_file["meta_idx"][index])
            metadata = self.metadata[meta_idx]
            sample_metadata = hdf5_file["sample_metadata"][index, ...][0]
            sample_metadata = eval(sample_metadata)
            if isinstance(metadata, np.ndarray) and len(metadata) == 1:
                metadata = metadata[0]
            elif isinstance(metadata, bytes):
                metadata = metadata.decode('UTF-8')
            try:
                metadata = eval(metadata)
                metadata.update(sample_metadata)
            except TypeError:
                pass # FI
            # where bandwise array has no data values, set as np.nan
            # sat_img[sat_img == metadata['nodata']] = np.nan # TODO: problem with lack of dynamic range. See: https://rasterio.readthedocs.io/en/latest/topics/masks.html

        sample = {"sat_img": sat_img, "map_img": map_img, "metadata": metadata,
                  "hdf5_path": self.hdf5_path}

        if self.radiom_transform:  # radiometric transforms should always precede geometric ones
            sample = self.radiom_transform(sample)
        if self.geom_transform:  # rotation, geometric scaling, flip and crop. Will also put channels first and convert to torch tensor from numpy.
            sample = self.geom_transform(sample)

        sample = self.totensor_transform(sample)

        if self.debug:
            # assert no new class values in map_img
            initial_class_ids = set(np.unique(map_img))
            final_class_ids = set(np.unique(sample['map_img'].numpy()))
            if not final_class_ids.issubset(initial_class_ids):
                logging.critical(f"\nWARNING: Class ids for label before and after augmentations don't match. "
                                 f"Ignore if overwritting ignore_index in ToTensorTarget")
        sample['index'] = index
        return sample
