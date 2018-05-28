"""
From:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import torch
from torch.utils.data import Dataset
import os
import numpy as np
# import entrainement_modele

class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation"""
    def __init__(self, workFolder, num_samples, sample_size):
        self.workFolder = workFolder
        self.num_samples = num_samples
        self.sample_size = sample_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        data_file = open(os.path.join(self.workFolder, "samples_RGB.dat"), "rb")
        ref_file = open(os.path.join(self.workFolder, "samples_Label.dat"), "rb")
        data_file.seek(index*self.sample_size*self.sample_size*3)
        ref_file.seek(index*self.sample_size*self.sample_size)

        data = np.float32(np.reshape(np.fromfile(data_file, dtype=np.uint8, count=3*self.sample_size*self.sample_size), [3, self.sample_size, self.sample_size]))
        target = np.int64(np.reshape(np.fromfile(ref_file, dtype=np.uint8, count=self.sample_size*self.sample_size), [self.sample_size, self.sample_size]))

        data_file.close()
        ref_file.close()

        sample = {'sat_img':data, 'map_img':target}
        return sample
