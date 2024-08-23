from typing import Any, Dict, List, Tuple
from kornia.augmentation import AugmentationSequential
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import kornia as K

from ..datasets import imagery_NonGeoDataset

class BlueSkyNonGeoDataModule(pl.LightningDataModule):
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
        
        self.normalize = K.augmentation.Normalize(mean=self.mean, std=self.std, p=1)
        random_resized_crop_zoom_in = K.augmentation.RandomResizedCrop(size=self.patch_size, scale=(1.0, 2.0), p=0.5)
        random_resized_crop_zoom_out = K.augmentation.RandomResizedCrop(size=self.patch_size, scale=(0.5, 1.0), p=0.5)
        
        
        self.transform = AugmentationSequential(K.augmentation.container.ImageSequential
                                                      (K.augmentation.RandomHorizontalFlip(p=0.5),
                                                       K.augmentation.RandomVerticalFlip(p=0.5),
                                                       K.augmentation.RandomAffine(degrees=[-45., 45.], p=0.5),
                                                       random_resized_crop_zoom_in,
                                                       random_resized_crop_zoom_out,
                                                       random_apply=1),
                                                      data_keys=["image", "label"]
                                                      )
    
    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["image"] /= self.data_type_max
        sample["image"] = self.normalize(sample["image"])
        return sample

    def prepare_data(self):
        # download, enhance, tile, etc...
        pass

    def setup(self, stage=None):
        # build the dataset
        train_transform = Compose([self.preprocess, self.transform])
        test_transform = Compose([self.preprocess])
        
        self.train_dataset = BlueSkyNonGeo(split="trn", transform=train_transform, **self.kwargs)
        self.val_dataset = BlueSkyNonGeo(split="val", transform=test_transform, **self.kwargs)
        self.test_dataset = BlueSkyNonGeo(split="tst", transform=test_transform, **self.kwargs)
        

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
    
    # print(len(dataset.train_dataset))
    # print(len(dataset.val_dataset))
    # print(len(dataset.test_dataset))
    # print(dataset.train_dataset._load_image(0).max())
    # print(dataset.train_dataset._load_label(0).max())
    # print(dataset.val_dataset._load_image(0).max())
    # print(dataset.val_dataset._load_label(0).max())
    # print(dataset.test_dataset._load_image(0).max())
    # print(dataset.test_dataset._load_label(0).max())
    # print(dataset.train_dataset[0]["image"].max())
    # print(dataset.train_dataset[0]["label"].max())
    # print(dataset.val_dataset[0]["image"].max())
    # print(dataset.val_dataset[0]["label"].max())
    # print(dataset.test_dataset[0]["image"].max())
    # print(dataset.test_dataset[0]["label"].max())
    # print(dataset.train_dataloader())
    # print(dataset.val_dataloader())
    # print(dataset.test_dataloader())
    # print(dataset.train_dataloader().dataset)
    # print(dataset.val_dataloader().dataset)
    # print(dataset.test_dataloader().dataset)
    # print(dataset.train_dataloader().dataset[0])
    # print(dataset.val_dataloader().dataset[0])
    # print(dataset.test_dataloader().dataset[0])
    # print(dataset.train_dataloader().dataset[0]["image"].max())
    # print(dataset.train_dataloader().dataset[0]["label"].max())
    # print(dataset.val_dataloader().dataset[0]["image"].max())
    # print(dataset.val_dataloader().dataset[0]["label"].max())
    # print(dataset.test_dataloader().dataset[0]["image"].max())
    # print(dataset.test_dataloader().dataset[0]["label"].max())
    # print(dataset.train_dataloader().dataset[0]["image"].shape)
    # print(dataset.train_dataloader().dataset[0]["label"].shape)
    # print(dataset.val_dataloader().dataset[0]["image"].shape)
    # print(dataset.val_dataloader().dataset[0]["label"].shape)
    # print(dataset.test_dataloader