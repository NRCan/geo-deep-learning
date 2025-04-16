import os
import torch
import pandas
import numpy as np
import rasterio as rio

from typing import Dict, List, Callable, Optional
from torch import Tensor
from torchgeo.datasets import NonGeoDataset


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.visualization import visualize_prediction


class SarWaterNonGeo(NonGeoDataset):
    """ 
    This dataset class is intended to handle data for semantic segmentation of geospatial imagery (Binary | Multiclass).
    It loads geospatial image and mask patches from csv files.
    
    Dataset format:
    
    * images are composed of arbitrary number of bands e.g. image_rgb_shape =  (224, 224, 3)
    * masks are single band images with pixel values representing classes e.g. mask_shape =  (224, 224, 1)
    * images and masks are stored in trn, val, and tst folders
    * csv files contain the path part starting to the image and mask pairs e.g. 'trn/image/0.tif;trn/label/0_lbl.tif'
    * csv files may contain additional columns for other metadata e.g. 'trn/image/0.tif;trn/label/0_lbl.tif;aoi_id'
    * csv files are named after the data split e.g. 'trn.csv', 'val.csv', 'tst.csv'
    
    root_directory
    ├───trn
    │   ├───image
    │   │       0.tif
    │   ├───label
    │           0_lbl.tif
    ├───val
    │   ├───image
    │   │       0.tif
    │   ├───label
    │           0_lbl.tif
    ├───tst
    │   ├───image
    │   │       0.tif
    │   ├───label
    │           0_lbl.tif
    ├───trn.csv
    ├───val.csv
    ├───tst.csv
    """
    
    def __init__(self, 
                 csv_root_folder: str,
                 patches_root_folder: str,
                 split: str = "trn",
                 transforms: Optional[Callable] = None
                 ):
        """Initialize the dataset.

        Args:
            csv_root_folder (str): The root folder where the csv files are stored
            patches_root_folder (str): The root folder where the image and mask patches are stored
            splits (List[str], optional): The data split. Defaults to ["trn", "val", "tst"].
        """
        self.csv_root_folder = csv_root_folder
        self.patches_root_folder = patches_root_folder
        if split == 'trn':
            split = 'trn_3x'
        self.split = split
        self.transforms = transforms
        self.files = self._load_files()
    
    def _load_files(self) -> List[Dict[str, str]]:
        """Load image and mask paths from csv files.

        Returns:
            List[Dict[str, str]]: list of dictionaries containing image and mask paths 
        """
        csv_path = os.path.join(self.csv_root_folder, f"{self.split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file {csv_path} not found.")
        df = pandas.read_csv(csv_path, header=None, sep=";")
        if len(df.columns) == 1:
            raise ValueError("CSV file must contain at least two columns: image_path;mask_path")
        
        files = [
            {"image": os.path.join(self.patches_root_folder, img),
             "mask": os.path.join(self.patches_root_folder, lbl)} 
            for img, lbl in df[[0, 1]].itertuples(index=False)
            ]
        
        return files
         
    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)
    
    def _load_image(self, index: int) -> Tensor:
        """ Load image

        Args:
            index (int): index of the image to load

        Returns:
            torch.Tensor: image tensor
        """
        image_path = self.files[index]["image"]
        image_name = os.path.basename(image_path)
        with rio.open(image_path) as image:
            image_array = image.read().astype(np.float32)
            image_tensor = torch.from_numpy(image_array)
            
        return image_tensor, image_name
    
    def _load_mask(self, index: int) -> Tensor:
        """ Load mask

        Args:
            index (int): index of the mask to load

        Returns:
            torch.Tensor: mask tensor
        """
        mask_path = self.files[index]["mask"]
        mask_name = os.path.basename(mask_path)
        with rio.open(mask_path) as mask:
            mask_array = mask.read().astype(np.int32)
            mask_tensor = torch.from_numpy(mask_array).float()

        return mask_tensor, mask_name
    
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return the image and mask tensors for the given index.

        Args:
            index (int): index of the sample to return
        
        Returns:
            Tuple[Tensor, Tensor]: image and mask tensors
        """
        image, image_name = self._load_image(index)
        mask, mask_name = self._load_mask(index)
        
        sample = {"image": image, "mask": mask}
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        sample["image_name"] = image_name
        sample["mask_name"] = mask_name
        return sample

if __name__ == "__main__":
    csv_root_folder = "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/DA_SAR_FLOOD/sar_water_extraction/data/datasets/Sentinel/patches/Sentinel"
    patches_root_folder = csv_root_folder
    dataset = SarWaterNonGeo(csv_root_folder, patches_root_folder, split="val")
    print(len(dataset))

    image, mask = dataset[1]["image"], dataset[1]["mask"]
    image = image.squeeze(0)
    mask = mask.squeeze(0)

    visualize_prediction(
        image=image,
        mask=mask,
        prediction=mask,
        num_classes=2,
        save_samples=False,
        class_colors=["#000000", "#0000FF"],
        save_path="/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/DA_SAR_FLOOD/sar_water_extraction/GDL/gdl_lightning/geo_deep_learning/datasets/samples"
    )