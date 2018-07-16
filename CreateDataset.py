"""
From:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import h5py
from torch.utils.data import Dataset
import os

class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation"""
    def __init__(self, workFolder, num_samples, dataset_type, transform = None):
        self.workFolder = workFolder
        self.num_samples = num_samples
        self.dataset_type = dataset_type
        self.transform = transform

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        
        hdf5_file = h5py.File(os.path.join(self.workFolder, self.dataset_type + "_samples.hdf5"), "r")
        
        sat_img = hdf5_file["sat_img"][index, ...]
        map_img = hdf5_file["map_img"][index, ...]

        hdf5_file.close()
        
        sample = {'sat_img':sat_img, 'map_img':map_img}

        if self.transform:
            sample = self.transform(sample)

        return sample