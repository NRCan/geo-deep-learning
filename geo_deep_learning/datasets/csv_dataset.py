"""CSV dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from geo_deep_learning.tools.utils import normalization, standardization

logger = logging.getLogger(__name__)


@rank_zero_only
def log_dataset(split: str, patch_count: int) -> None:
    """Log dataset."""
    logger.info("Created dataset for %s split with %s patches", split, patch_count)


class CSVDataset(NonGeoDataset):
    """

    Loads geospatial image and mask patches from csv files.

    Dataset format:

    * images are composed of arbitrary number of bands
    * masks are single band images with pixel values representing classes
    * images and masks are stored in trn, val, and tst folders

    * csv files contain the path part starting to the image and mask pairs
        e.g. 'trn/image/0.tif;trn/label/0_lbl.tif'

    * csv files may contain additional columns for other metadata
        e.g. 'trn/image/0.tif;trn/label/0_lbl.tif;aoi_id'

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
            patches_root_folder (str): The root folder of image and mask patches.
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
        """Load image and mask paths from csv files."""
        csv_path = Path(self.csv_root_folder) / f"{self.split}.csv"
        if not csv_path.exists():
            msg = f"CSV file {csv_path} not found."
            raise FileNotFoundError(msg)
        df_csv = pd.read_csv(csv_path, header=None, sep=";")
        if len(df_csv.columns) == 1:
            msg = "CSV file must contain at least two columns: image_path;mask_path"
            raise ValueError(msg)

        return [
            {
                "image": Path(self.patches_root_folder) / img,
                "mask": Path(self.patches_root_folder) / lbl,
            }
            for img, lbl in df_csv[[0, 1]].itertuples(index=False)
        ]

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

        return image_tensor, image_name

    def _load_mask(self, index: int) -> Tensor:
        """Load mask."""
        mask_path = self.files[index]["mask"]
        mask_name = Path(mask_path).name
        with rio.open(mask_path) as mask:
            mask_array = mask.read().astype(np.int32)
            mask_tensor = torch.from_numpy(mask_array).float()

        return mask_tensor, mask_name

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Return the image and mask tensors for the given index.

        Args:
            index (int): index of the sample to return

        Returns:
            Tuple[Tensor, Tensor]: image and mask tensors

        """
        image, image_name = self._load_image(index)
        image = normalization(image)
        mean = torch.tensor(self.norm_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.norm_stats["std"], dtype=torch.float32).view(-1, 1, 1)
        image = standardization(image, mean, std)
        mask, mask_name = self._load_mask(index)

        sample = {"image": image, "mask": mask}

        sample["image_name"] = image_name
        sample["mask_name"] = mask_name
        sample["mean"] = mean
        sample["std"] = std
        return sample


if __name__ == "__main__":
    csv_root_folder = ""
    patches_root_folder = csv_root_folder
    dataset = CSVDataset(csv_root_folder, patches_root_folder, split="val")
    # print(len(dataset))
    # print(dataset._load_image(0).max())
    # print(dataset._load_mask(0).max())
