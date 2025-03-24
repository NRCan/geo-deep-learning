import sys
import torch
import numpy as np
from typing import Optional, Any, Dict, List, Tuple
import kornia.augmentation as K
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from pathlib import Path
import warnings

from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.SAR_FloodAsWaterDataset import SarWaterNonGeo
from tools.utils import normalization, standardization

class SarWaterNonGeoDataModule(LightningDataModule):
    
    def __init__(self, 
                 batch_size: int = 16,
                 num_workers: int = 8,
                 data_type_max: int = 255,
                 patch_size: Tuple[int, int] = (512, 512),
                 mean: List[float] = [0.0, 0.0, 0.0],
                 std: List[float] = [1.0, 1.0, 1.0],
                 band_indices: Optional[List[int]] = None, 
                 noData: int = 255,
                 **kwargs: Any):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.data_type_max = data_type_max
        self.mean = mean
        self.std = std
        self.band_indices = band_indices
        self.noData = noData
        self.kwargs = kwargs
        
        if self.band_indices is not None:
            self.mean = [self.mean[i] for i in self.band_indices]
            self.std = [self.std[i] for i in self.band_indices]
        
        self.normalize = K.Normalize(mean=self.mean, std=self.std, p=1, keepdim=True)
        random_resized_crop_zoom_in = K.RandomResizedCrop(size=self.patch_size, scale=(1.0, 2.0), 
                                                                       p=0.5, align_corners=False, keepdim=True)
        random_resized_crop_zoom_out = K.RandomResizedCrop(size=self.patch_size, scale=(0.5, 1.0), 
                                                                        p=0.5, align_corners=False, keepdim=True)
        
        # self.transform = K.AugmentationSequential(K.RandomHorizontalFlip(p=0.5, keepdim=True),
        #                                         K.RandomVerticalFlip(p=0.5, keepdim=True),
        #                                         K.RandomRotation90(times=(1, 3), 
        #                                                                     p=0.5, 
        #                                                                     align_corners=True, 
        #                                                                     keepdim=True),
        #                                         random_resized_crop_zoom_in,
        #                                         random_resized_crop_zoom_out,
        #                                         data_keys=None,
        #                                         random_apply=1)
    def prepare_data(self):
        # download, enhance, tile, etc...
        pass
    
    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["image"] = self._manage_bands(sample["image"])
        sample = self._convert_noData(sample)
        # sample["image"] /= self.data_type_max
        sample["image"] = self.normalize(sample["image"])
        
        return sample

    def setup(self, stage=None):
        # build the dataset
        #train_transform = Compose([self.transform, self.preprocess])
        train_transform = Compose([self.preprocess])
        test_transform = Compose([self.preprocess])
        
        self.train_dataset = SarWaterNonGeo(split="trn", transforms=train_transform, **self.kwargs)
        self.val_dataset = SarWaterNonGeo(split="val", transforms=test_transform, **self.kwargs)
        self.test_dataset = SarWaterNonGeo(split="tst", transforms=test_transform, **self.kwargs)

        # print("Calculating mean and standard deviation for the dataset...")
        # mean, std = self.calculate_dataset_mean_std(self.train_dataset)
        # print(f"Mean: {mean}")
        # print(f"Standard deviation: {std}")


    def calculate_dataset_mean_std(self, dataset):
        """
        Calculate the mean and standard deviation for each band of the dataset
        
        Args:
            dataset: The dataset for which to calculate the mean and standard deviation.
            
            Returns:
            mean: List of mean values for each band
            std: List of standard deviation values for each band
            
        """

        mean = torch.zeros(len(dataset[0]["image"]))
        std = torch.zeros(len(dataset[0]["image"]))

        for sample in tqdm(dataset):
            image = sample["image"]
            mean += image.mean(dim=[1, 2])
            std += image.std(dim=[1, 2])

        mean /= len(dataset)
        std /= len(dataset)

        return mean, std

    def train_dataloader(self) -> DataLoader[Any]:
        
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          prefetch_factor=2,
                          shuffle=True)
       
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          prefetch_factor=2,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          prefetch_factor=2,
                          shuffle=False)
    
    def _manage_bands(self, image: torch.Tensor) -> torch.Tensor:
        """
        Selects specific bands from the input image tensor based on predefined band indices.

        Args:
            image (torch.Tensor): Input tensor of shape [C, H, W], where C is the number of bands.

        Returns:
            torch.Tensor: Tensor containing only the selected bands.

        Raises:
            ValueError: If any band index in `self.band_indices` is out of range for the input image.
        """
        if self.band_indices is not None:
            bands = image.size(0)
            if max(self.band_indices) >= bands:
                raise ValueError(f"Band index {max(self.band_indices)} is out of range for image with {bands} bands")
            band_indices = torch.LongTensor(self.band_indices).to(image.device)
            return torch.index_select(image, dim=0, index=band_indices)
        return image
    
    def _model_script_preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses the input sample by normalizing and standardizing the image.
        Args:
            sample (Dict[str, Any]): A dictionary containing the input data. 
                Expected to have a key "image" with a tensor value.
        Returns:
            Dict[str, Any]: The processed sample with the image normalized and standardized.
        """
        device = sample["image"].device
        mean_ = torch.tensor(self.mean, device=device).reshape(len(self.mean), 1)
        std_ = torch.tensor(self.std, device=device).reshape(len(self.std), 1)
        
        # sample["image"] = normalization(sample["image"], 0, self.data_type_max)
        sample["image"] = standardization(sample["image"], mean_, std_)
        return sample
    
    def _convert_noData(self, sample: Dict[str, any]) -> Dict[str, any]:
        """
        Converts the noData value in the input sample to 0.
        Args:
            sample (Dict[str, any]): A dictionary containing the input data. 
                Expected to have a key "image" with a tensor value.
        Returns:
            Dict[str, any]: The processed sample with the noData value converted to NaN.
        """
        sample["image"][sample["image"] == self.noData] = 0
        sample["mask"][sample["mask"] == self.noData] = 0
        return sample

    def _compute_class_weights(self, class_frequency):
        """
        Computes class weights inversely proportional to class frequencies.

        Args:
            class_frequency: A dictionary with class labels as keys and their corresponding
                            frequency counts as values.

        Returns:
            class_weights: A dictionary with class labels as keys and their corresponding
                        weights as values.
        """
        total_pixels = sum(class_frequency.values())
        num_classes = len(class_frequency)
        class_weights = {}

        for cls, freq in class_frequency.items():
            # Inverse frequency
            class_weight = total_pixels / (num_classes * freq)
            class_weights[cls] = class_weight

        return class_weights
    
    def _compute_class_frequency(self, dataset):
        """
        Computes the frequency of each class in the input dataset.

        Args:
            dataset: The dataset for which to compute the class frequency. 
                    Assumes each item in the dataset is a dictionary with a 'mask' key.

        Returns:
            class_frequency: A dictionary with class labels as keys and their corresponding
                            frequency counts as values.
        """
        class_frequency = {}

        for sample in dataset:
            mask = sample["mask"].numpy()  # Convert tensor to numpy array if necessary
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                if cls in class_frequency:
                    class_frequency[cls] += count
                else:
                    class_frequency[cls] = count

        return class_frequency

if __name__ == "__main__":
    config = {
        "batch_size": 16,
        "num_workers": 8,
        "data_type_max": 1,
        "noData": 255,
        "patch_size": [512, 512],
        "mean": [0.0, 0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0, 1.0],
        "csv_root_folder": "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/DA_SAR_FLOOD/sar_water_extraction/data/datasets/Sentinel/patches/Sentinel",
        "patches_root_folder": "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/DA_SAR_FLOOD/sar_water_extraction/data/datasets/Sentinel/patches/Sentinel"
    }
    dataset = SarWaterNonGeoDataModule(**config)
    dataset.setup()
    # tst = dataset.test_dataloader()
    # for batch in tst:
    #     print(batch["image"])
    #     break