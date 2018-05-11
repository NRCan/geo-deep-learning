""" from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html """
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import entrainement_modele

class SegmentationDataset(Dataset):
    """Dataset pour faire la segmentation semantique"""
    def __init__(self, workFolder, nbrEchantillons, Taille_tuile):
        self.workFolder = workFolder
        self.nbrEchant = nbrEchantillons
        self.tailleTuile = Taille_tuile
        
    def __len__(self):
        return self.nbrEchant
    
    def __getitem__(self, index):
        
        data_file = open(os.path.join(self.workFolder, "echantillons_RGB.dat"), "rb")
        ref_file = open(os.path.join(self.workFolder, "echantillons_Label.dat"), "rb")   
        data_file.seek(index*self.tailleTuile*self.tailleTuile*3)
        ref_file.seek(index*self.tailleTuile*self.tailleTuile)
        
        data = np.float32(np.reshape(np.fromfile(data_file, dtype=np.uint8, count=3*self.tailleTuile*self.tailleTuile), [3, self.tailleTuile, self.tailleTuile]))
        target = np.int64(np.reshape(np.fromfile(ref_file, dtype=np.uint8, count=self.tailleTuile*self.tailleTuile), [self.tailleTuile, self.tailleTuile]))
        data_n = np.reshape(np.fromfile(data_file, dtype=np.uint8, count=3*self.tailleTuile*self.tailleTuile), [3, self.tailleTuile, self.tailleTuile])
        target_n = np.reshape(np.fromfile(ref_file, dtype=np.uint8, count=self.tailleTuile*self.tailleTuile), [self.tailleTuile, self.tailleTuile])
        
        # print(data_n.shape)
        # print(target_n.shape)
        # print(index)
        # entrainement_modele.plot_some_results(data_n, target_n, index, self.workFolder)
        # val = target.ravel()
        # print(max(val),min(val))
        
        data_file.close()
        ref_file.close()
        TorchData = torch.from_numpy(data)
        TorchTarget = torch.from_numpy(target)
        del data
        del target
        
        sample = {'sat_img':TorchData, 'map_img':TorchTarget}
        return sample