import torch
from typing import Any, Dict, List, Tuple
from kornia.augmentation import AugmentationSequential
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from tools.utils import normalization, standardization
import kornia as K
from datasets.imagery_NonGeoDataset import BlueSkyNonGeo

class BlueSkyNonGeoDataModule(LightningDataModule):
    def __init__(self, 
                 batch_size: int = 16,
                 num_workers: int = 8,
                 data_type_max: int = 255,
                 patch_size: Tuple[int, int] = (512, 512),
                 mean: List[float] = [0.0, 0.0, 0.0],
                 std: List[float] = [1.0, 1.0, 1.0],
                 **kwargs: Any):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.data_type_max = data_type_max
        self.mean = mean
        self.std = std
        self.kwargs = kwargs
        
        self.normalize = K.augmentation.Normalize(mean=self.mean, std=self.std, p=1, keepdim=True)
        random_resized_crop_zoom_in = K.augmentation.RandomResizedCrop(size=self.patch_size, scale=(1.0, 2.0), 
                                                                       p=0.5, keepdim=True)
        random_resized_crop_zoom_out = K.augmentation.RandomResizedCrop(size=self.patch_size, scale=(0.5, 1.0), 
                                                                        p=0.5, keepdim=True)
        
        
        self.transform = AugmentationSequential(K.augmentation.container.ImageSequential
                                                      (K.augmentation.RandomHorizontalFlip(p=0.5, 
                                                                                           keepdim=True),
                                                       K.augmentation.RandomVerticalFlip(p=0.5, 
                                                                                         keepdim=True),
                                                       K.augmentation.RandomAffine(degrees=[-45., 45.], 
                                                                                   p=0.5, 
                                                                                   keepdim=True),
                                                       random_resized_crop_zoom_in,
                                                       random_resized_crop_zoom_out,
                                                       random_apply=1), data_keys=None
                                                      )
    def model_script_preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        device = sample["image"].device
        mean_ = torch.tensor(self.mean, device=device).reshape(len(self.mean), 1)
        std_ = torch.tensor(self.std, device=device).reshape(len(self.std), 1)
        
        sample["image"] = normalization(sample["image"], 0, self.data_type_max)
        sample["image"] = standardization(sample["image"], mean_, std_)
        return sample
    
    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["image"] /= self.data_type_max
        sample["image"] = self.normalize(sample["image"])
        
        return sample

    def prepare_data(self):
        # download, enhance, tile, etc...
        pass

    def setup(self, stage=None):
        # build the dataset
        train_transform = Compose([self.transform, self.preprocess])
        test_transform = Compose([self.preprocess])
        
        self.train_dataset = BlueSkyNonGeo(split="trn", transforms=train_transform, **self.kwargs)
        self.val_dataset = BlueSkyNonGeo(split="val", transforms=test_transform, **self.kwargs)
        self.test_dataset = BlueSkyNonGeo(split="tst", transforms=test_transform, **self.kwargs)

    def train_dataloader(self) -> DataLoader[Any]:
        
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True)
       
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
    

if __name__ == "__main__":
    csv_root_folder = "/export/sata01/wspace/test_dir/single/hydro/clahe_NRG/tst_ecozones/hydro_all_ecozones"
    # csv_root_folder = "/export/sata01/wspace/test_dir/multi/all_rgb_data/patches/4cls_RGB"
    patches_root_folder = csv_root_folder
    dataset = BlueSkyNonGeoDataModule(csv_root_folder, patches_root_folder, split="val")
    print(f"mean:{dataset.mean}, std:{dataset.std}")