import os
import h5py
from torch.utils.data import Dataset
import numpy as np


def create_files_and_datasets(params, samples_folder):
    """
    Function to create the hdfs files (trn, val and tst).
    :param params: (dict) Parameters found in the yaml config file.
    :param samples_folder: (str) Path to the output folder.
    :return: (hdf5 datasets) trn, val ant tst datasets.
    """
    samples_size = params['global']['samples_size']
    number_of_bands = params['global']['number_of_bands']

    trn_hdf5 = h5py.File(os.path.join(samples_folder, "trn_samples.hdf5"), "w")
    val_hdf5 = h5py.File(os.path.join(samples_folder, "val_samples.hdf5"), "w")
    tst_hdf5 = h5py.File(os.path.join(samples_folder, "tst_samples.hdf5"), "w")

    trn_hdf5.create_dataset("sat_img", (0, samples_size, samples_size, number_of_bands), np.float32,
                            maxshape=(None, samples_size, samples_size, number_of_bands))
    trn_hdf5.create_dataset("map_img", (0, samples_size, samples_size), np.int16,
                            maxshape=(None, samples_size, samples_size))
    trn_hdf5.create_dataset("meta_idx", (0, 1), dtype=np.int16, maxshape=(None, 1))
    trn_hdf5.create_dataset("metadata", (0, 1), dtype=h5py.string_dtype(), maxshape=(None, 1))

    val_hdf5.create_dataset("sat_img", (0, samples_size, samples_size, number_of_bands), np.float32,
                            maxshape=(None, samples_size, samples_size, number_of_bands))
    val_hdf5.create_dataset("map_img", (0, samples_size, samples_size), np.int16,
                            maxshape=(None, samples_size, samples_size))
    val_hdf5.create_dataset("meta_idx", (0, 1), dtype=np.int16, maxshape=(None, 1))
    val_hdf5.create_dataset("metadata", (0, 1), dtype=h5py.string_dtype(), maxshape=(None, 1))

    tst_hdf5.create_dataset("sat_img", (0, samples_size, samples_size, number_of_bands), np.float32,
                            maxshape=(None, samples_size, samples_size, number_of_bands))
    tst_hdf5.create_dataset("map_img", (0, samples_size, samples_size), np.int16,
                            maxshape=(None, samples_size, samples_size))
    tst_hdf5.create_dataset("meta_idx", (0, 1), dtype=np.int16, maxshape=(None, 1))
    tst_hdf5.create_dataset("metadata", (0, 1), dtype=h5py.string_dtype(), maxshape=(None, 1))
    return trn_hdf5, val_hdf5, tst_hdf5


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation"""
    def __init__(self, work_folder, dataset_type, max_sample_count=None, transform=None):
        # note: if 'max_sample_count' is None, then it will be read from the dataset at runtime
        self.work_folder = work_folder
        self.max_sample_count = max_sample_count
        self.dataset_type = dataset_type
        self.transform = transform
        self.metadata = []
        self.hdf5_path = os.path.join(self.work_folder, self.dataset_type + "_samples.hdf5")
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            if "metadata" in hdf5_file:
                for i in range(hdf5_file["metadata"].shape[0]):
                    metadata = hdf5_file["metadata"][i, ...]
                    if isinstance(metadata, np.ndarray) and len(metadata) == 1:
                        metadata = metadata[0]
                    self.metadata.append(metadata)
            if self.max_sample_count is None:
                self.max_sample_count = hdf5_file["sat_img"].shape[0]

    def __len__(self):
        return self.max_sample_count

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            sat_img = hdf5_file["sat_img"][index, ...]
            map_img = hdf5_file["map_img"][index, ...]
            meta_idx = int(hdf5_file["meta_idx"][index]) if "meta_idx" in hdf5_file else -1
            metadata = None
            if meta_idx != -1:
                metadata = self.metadata[meta_idx]

        sample = {'sat_img': sat_img, 'map_img': map_img, 'metadata': metadata}

        if self.transform:
            sample = self.transform(sample)

        return sample
