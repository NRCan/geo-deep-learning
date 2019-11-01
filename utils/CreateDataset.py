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

    trn_hdf5 = h5py.File(os.path.join(samples_folder, "trn_samples.hdf5"), "a")
    val_hdf5 = h5py.File(os.path.join(samples_folder, "val_samples.hdf5"), "a")
    tst_hdf5 = h5py.File(os.path.join(samples_folder, "tst_samples.hdf5"), "a")

    #if not trn_hdf5.mode=='r+':    #FIXME "Unable to open object (object 'sat_img' doesn't exist)"
    trn_hdf5.create_dataset("sat_img", (0, samples_size, samples_size, number_of_bands), np.float32,
                            maxshape=(None, samples_size, samples_size, number_of_bands))
    trn_hdf5.create_dataset("map_img", (0, samples_size, samples_size), np.uint8,
                            maxshape=(None, samples_size, samples_size))

    val_hdf5.create_dataset("sat_img", (0, samples_size, samples_size, number_of_bands), np.float32,
                            maxshape=(None, samples_size, samples_size, number_of_bands))
    val_hdf5.create_dataset("map_img", (0, samples_size, samples_size), np.uint8,
                            maxshape=(None, samples_size, samples_size))

    tst_hdf5.create_dataset("sat_img", (0, samples_size, samples_size, number_of_bands), np.float32,
                            maxshape=(None, samples_size, samples_size, number_of_bands))
    tst_hdf5.create_dataset("map_img", (0, samples_size, samples_size), np.uint8,
                            maxshape=(None, samples_size, samples_size))

    return trn_hdf5, val_hdf5, tst_hdf5


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation"""
    def __init__(self, work_folder, num_samples, dataset_type, transform=None):
        self.work_folder = work_folder
        self.num_samples = num_samples
        self.dataset_type = dataset_type
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        hdf5_file = h5py.File(os.path.join(self.work_folder, self.dataset_type + "_samples.hdf5"), "r")

        sat_img = hdf5_file["sat_img"][index, ...]
        map_img = hdf5_file["map_img"][index, ...]

        hdf5_file.close()

        sample = {'sat_img': sat_img, 'map_img': map_img}

        if self.transform:
            sample = self.transform(sample)

        return sample
