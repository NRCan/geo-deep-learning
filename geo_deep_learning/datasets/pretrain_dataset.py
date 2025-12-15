"""Pretrain dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from geo_deep_learning.utils.tensors import normalization

logger = logging.getLogger(__name__)


@rank_zero_only
def log_dataset(split: str, patch_count: int) -> None:
    """Log dataset."""
    logger.info("Created dataset for %s split with %s patches", split, patch_count)


class PretrainDataset(NonGeoDataset):
    """

    Loads geospatial image patches from csv files for pretraining (unsupervised learning).

    Dataset format:

    * images are composed of arbitrary number of bands
    * images are stored in trn and tst folders

    * csv files contain single column with image paths: 'trn/0.tif'

    * csv files are named after the data split e.g. 'trn.csv', 'tst.csv'

    root_directory
    ├───trn
    │       0.tif
    │       1.tif
    ├───tst
    │       0.tif
    │       1.tif
    ├───trn.csv
    ├───tst.csv
    """

    def __init__(
        self,
        csv_root_folder: str,
        patches_root_folder: str,
        split: str = "trn",
        norm_stats: dict[str, list[float]] | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            csv_root_folder (str): The root folder where the csv files are stored
            patches_root_folder (str): The root folder of image patches.
            split (str, optional): Defaults to "trn".
            norm_stats (dict[str, list[float]], optional): Normalization statistics.

        """
        self.csv_root_folder = csv_root_folder
        self.patches_root_folder = patches_root_folder
        self.split = split
        self.norm_stats = norm_stats
        self.files = self._load_files()
        log_dataset(self.split, len(self.files))

    def _load_files(self) -> list[dict[str, str]]:
        """Load image paths from csv files."""
        csv_path = Path(self.csv_root_folder) / f"{self.split}.csv"
        if not csv_path.exists():
            msg = f"CSV file {csv_path} not found."
            raise FileNotFoundError(msg)
        df_csv = pd.read_csv(csv_path, header=None, sep=";")
        
        files = []
        for row in df_csv.itertuples(index=False):
            # Pretraining mode: only image path (single column)
            image_path = row[0]
            files.append({
                "image": Path(self.patches_root_folder) / image_path,
            })
        
        return files

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            length of the dataset

        """
        return len(self.files)

    def _load_image(self, index: int) -> Tensor:
        """Load image."""
        image_path = self.files[index]["image"]
        image_name = Path(image_path).name
        with rio.open(image_path) as image:
            image_array = image.read().astype(np.int32)
            image_tensor = torch.from_numpy(image_array).float()
        
        # Extract first 3 channels if image has more than 3 channels (for RGB models)
        if image_tensor.shape[0] > 3:
            image_tensor = image_tensor[:3]

        return image_tensor, image_name

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Return the image tensor for the given index.

        Args:
            index (int): index of the sample to return

        Returns:
            dict: Dictionary containing image tensor

        """
        image, image_name = self._load_image(index)
        image = normalization(image)

        sample = {"image": image, "image_name": image_name}

        return sample


if __name__ == "__main__":
    csv_root_folder = ""
    patches_root_folder = csv_root_folder
    dataset = PretrainDataset(csv_root_folder, patches_root_folder, split="trn")

