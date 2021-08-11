import collections
import logging
import os
import warnings
from typing import List

import h5py
from torch.utils.data import Dataset
import numpy as np

import models.coordconv
from utils.utils import get_key_def, ordereddict_eval, compare_config_yamls
from utils.geoutils import get_key_recursive

# These two import statements prevent exception when using eval(metadata) in SegmentationDataset()'s __init__()
from rasterio.crs import CRS
from affine import Affine

logging.getLogger(__name__)


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


def create_files_and_datasets(samples_size: int, number_of_bands: int, meta_map, samples_folder: str, params):
    """
    Function to create the hdfs files (trn, val and tst).
    :param samples_size: size of individual hdf5 samples to be created
    :param number_of_bands: number of bands in imagery
    :param meta_map:
    :param samples_folder: (str) Path to the output folder.
    :param params: (dict) Parameters found in the yaml config file.
    :return: (hdf5 datasets) trn, val ant tst datasets.
    """
    real_num_bands = number_of_bands - MetaSegmentationDataset.get_meta_layer_count(meta_map)
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
            append_to_dataset(hdf5_file["params"], repr(params))
        except AttributeError:
            logging.exception(f'Update h5py to version 2.10 or higher')
            raise
        hdf5_files.append(hdf5_file)

    if params['global']['qgis_tracker']: # change to get_key_def() to allow for yamls without the qgis_tracker term
        hdf5_file = h5py.File(os.path.join(samples_folder, f"tracker.hdf5"), "w")
        for subset in ["trn", "val", "tst"]:
            grp = hdf5_file.create_group(subset)
            grp.create_dataset("coords", (0, 4), dtype=float, maxshape=(None, 4))
            grp.create_dataset("projection", (0, 1), dtype=h5py.string_dtype(), maxshape=(None, 1))
        hdf5_files.append(hdf5_file)
    else:
        hdf5_files.append(None)

    return hdf5_files


class SegmentationDataset(Dataset):
    """Semantic segmentation dataset based on HDF5 parsing."""

    def __init__(self, work_folder,
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
        self.work_folder = work_folder
        self.max_sample_count = max_sample_count
        self.dataset_type = dataset_type
        self.num_bands = num_bands
        self.metadata = []
        self.radiom_transform = radiom_transform
        self.geom_transform = geom_transform
        self.totensor_transform = totensor_transform
        self.debug = debug
        self.dontcare = dontcare
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

            if dataset_type == 'trn' and isinstance(hdf5_params, dict) and isinstance(metadata, dict):
                # check match between current yaml and sample yaml for crucial parameters
                try:
                    compare_config_yamls(hdf5_params, params)
                except TypeError:
                    logging.exception("Couldn't compare current yaml with hdf5 yaml")

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
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            sat_img = np.float32(hdf5_file["sat_img"][index, ...])
            assert self.num_bands <= sat_img.shape[-1]
            map_img = self._remap_labels(hdf5_file["map_img"][index, ...])
            meta_idx = int(hdf5_file["meta_idx"][index])
            metadata = self.metadata[meta_idx]
            sample_metadata = hdf5_file["sample_metadata"][index, ...][0]
            try:
                sample_metadata = eval(sample_metadata.decode('UTF-8'))
            except AttributeError:
                pass # TODO: hdf5_file["sample_metadata"] = wha?
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
        # TODO: figure out this class's """ERROR 1: PROJ: proj_create_from_name: Cannot find proj.db""" it spams when debugging & loading self variable
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


class MetaSegmentationDataset(SegmentationDataset):
    """Semantic segmentation dataset interface that appends metadata under new tensor layers."""

    metadata_handling_modes = ["const_channel", "scaled_channel"]

    def __init__(self, work_folder,
                 dataset_type,
                 num_bands,
                 meta_map,
                 max_sample_count=None,
                 dontcare=None,
                 radiom_transform=None,
                 geom_transform=True,
                 totensor_transform=True,
                 debug=False):
        assert meta_map is None or isinstance(meta_map, dict), "unexpected metadata mapping object type"
        assert meta_map is None or all([isinstance(k, str) and v in self.metadata_handling_modes for k, v in meta_map.items()]), \
            "unexpected metadata key type or value handling mode"
        super().__init__(work_folder=work_folder, dataset_type=dataset_type, num_bands=num_bands,
                         max_sample_count=max_sample_count,
                         dontcare=dontcare,
                         radiom_transform=radiom_transform,
                         geom_transform=geom_transform,
                         totensor_transform=totensor_transform,
                         debug=debug)
        assert all([isinstance(m, (dict, collections.OrderedDict)) for m in self.metadata]), \
            "cannot use provided metadata object type with meta-mapping dataset interface"
        self.meta_map = meta_map

    @staticmethod
    def append_meta_layers(tensor, meta_map, metadata):
        if meta_map:
            assert isinstance(metadata, (dict, collections.OrderedDict)), "unexpected metadata type"
            for meta_key, mode in meta_map.items():
                meta_val = get_key_recursive(meta_key, metadata)
                if mode == "const_channel":
                    assert np.isscalar(meta_val), "constant channel-wise assignment requires scalar value"
                    layer = np.full(tensor.shape[0:2], meta_val, dtype=np.float32)
                    tensor = np.insert(tensor, tensor.shape[2], layer, axis=2)
                elif mode == "scaled_channel":
                    assert np.isscalar(meta_val), "scaled channel-wise coords assignment requires scalar value"
                    layers = models.coordconv.get_coords_map(tensor.shape[0], tensor.shape[1]) * meta_val
                    tensor = np.insert(tensor, tensor.shape[2], layers, axis=2)
                # else...
        return tensor

    @staticmethod
    def get_meta_layer_count(meta_map):
        meta_layers = 0
        if meta_map:
            for meta_key, mode in meta_map.items():
                if mode == "const_channel":
                    meta_layers += 1
                elif mode == "scaled_channel":
                    meta_layers += 2
        return meta_layers

    def __getitem__(self, index):
        # put metadata layer in util func for inf script?
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            sat_img = hdf5_file["sat_img"][index, ...]
            assert self.num_bands <= sat_img.shape[-1]
            map_img = self._remap_labels(hdf5_file["map_img"][index, ...])
            meta_idx = int(hdf5_file["meta_idx"][index])
            metadata = self.metadata[meta_idx]
            sample_metadata = hdf5_file["sample_metadata"][index, ...]
            if isinstance(metadata, np.ndarray) and len(metadata) == 1:
                metadata = metadata[0]
                sample_metadata = sample_metadata[0]
            if isinstance(metadata, str):
                metadata = eval(metadata)
                sample_metadata = eval(sample_metadata)
            metadata.update(sample_metadata)
            assert meta_idx != -1, f"metadata unvailable in sample #{index}"
            sat_img = self.append_meta_layers(sat_img, self.meta_map, self.metadata[meta_idx])
        sample = {"sat_img": sat_img, "map_img": map_img, "metadata": metadata}
        if self.radiom_transform:  # radiometric transforms should always precede geometric ones
            sample = self.radiom_transform(sample)  # TODO: test this for MetaSegmentationDataset
        sample["sat_img"] = self.append_meta_layers(sat_img, self.meta_map, metadata)  # Overwrite sat_img with sat_img with metalayers
        if self.geom_transform:
            sample = self.geom_transform(sample)  # rotation, geometric scaling, flip and crop. Will also put channels first and convert to torch tensor from numpy.
        sample = self.totensor_transform(sample)  # TODO: test this for MetaSegmentationDataset
        return sample