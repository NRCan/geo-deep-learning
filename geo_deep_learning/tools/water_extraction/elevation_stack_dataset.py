"""Elevation Stack dataset that extends CSVDataset for water extraction tasks."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from torch import Tensor

from geo_deep_learning.datasets.csv_dataset import CSVDataset, log_dataset
from geo_deep_learning.utils.tensors import normalization, standardization

logger = logging.getLogger(__name__)

# Constants
MIN_SPATIAL_DIMS = 2


class ElevationStackDataset(CSVDataset):
    """
    Dataset for elevation stack data (TWI, nDSM, intensity) with water body labels.

    Extends CSVDataset to handle CSV files with custom column names ("tif", "gpkg")
    and split information in a single CSV file rather than separate files per split.

    Expected CSV format:
        tif,gpkg,aoi,split
        /path/to/input.tif,/path/to/label.tif,aoi_name,trn
        /path/to/input2.tif,/path/to/label2.tif,aoi_name,val
        ...
    """

    def __init__(
        self,
        csv_root_folder: str,
        patches_root_folder: str,
        split: str = "trn",
        norm_stats: dict[str, list[float]] | None = None,
    ) -> None:
        """
        Initialize the ElevationStackDataset.

        Args:
            csv_root_folder (str): The root folder where the csv file is stored
            patches_root_folder (str): The root folder of image and mask patches.
            split (str, optional): Data split ("trn", "val", "tst"). Defaults to "trn".
            norm_stats (dict[str, list[float]], optional): Normalization statistics.
                Should contain "mean" and "std" keys with lists of per-band values.

        """
        # Initialize parent class attributes first
        self.csv_root_folder = csv_root_folder
        self.patches_root_folder = patches_root_folder
        self.split = split
        self.norm_stats = norm_stats or {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        }

        # Load files using custom method
        self.files = self._load_files()

        # Log dataset creation (using the same pattern as base class)
        log_dataset(self.split, len(self.files))

    def _load_files(self) -> list[dict[str, str]]:
        """
        Load image and mask paths from CSV file with custom column names.

        Overrides the base class method to handle CSV with "tif", "gpkg", and "split" columns.

        Returns:
            List of dictionaries with "image" and "mask" keys containing file paths.

        """
        # Look for a single CSV file in the root folder
        csv_files = list(Path(self.csv_root_folder).glob("*.csv"))
        if not csv_files:
            msg = f"No CSV files found in {self.csv_root_folder}"
            raise FileNotFoundError(msg)

        # Use the first CSV file found (or you could look for a specific name)
        csv_path = csv_files[0]
        logger.info("Loading dataset from: %s", csv_path)

        # Read CSV with headers
        df_csv = pd.read_csv(csv_path)

        # Validate required columns
        required_cols = ["tif", "gpkg", "split"]
        missing_cols = [col for col in required_cols if col not in df_csv.columns]
        if missing_cols:
            msg = (
                f"CSV file must contain columns: {required_cols}. "
                f"Missing: {missing_cols}"
            )
            raise ValueError(msg)

        # Filter by split
        split_df = df_csv[df_csv["split"] == self.split]
        if len(split_df) == 0:
            msg = f"No data found for split '{self.split}' in {csv_path}"
            raise ValueError(msg)

        logger.info("Found %d samples for split '%s'", len(split_df), self.split)

        # Convert to the format expected by base class
        return [
            {
                "image": Path(row["tif"]),  # Use absolute paths from CSV
                "mask": Path(row["gpkg"]),  # Use absolute paths from CSV
            }
            for _, row in split_df.iterrows()
        ]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Return the image and mask tensors for the given index.

        Extends base class to handle nodata values appropriately for elevation data.

        Args:
            index (int): index of the sample to return

        Returns:
            dict containing image, mask, and metadata tensors

        """
        image, image_name = self._load_image(index)
        mask, mask_name = self._load_mask(index)

        # Apply normalization (0-1 scaling)
        image = normalization(image)

        # Apply standardization using provided statistics
        mean = torch.tensor(self.norm_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.norm_stats["std"], dtype=torch.float32).view(-1, 1, 1)
        image = standardization(image, mean, std)

        # Handle mask - ensure it's long type for segmentation
        # Remove channel dim if present
        mask = mask.squeeze(0) if mask.dim() > MIN_SPATIAL_DIMS else mask
        mask = mask.long()

        sample = {
            "image": image,
            "mask": mask,
            "image_name": image_name,
            "mask_name": mask_name,
            "mean": mean,
            "std": std,
        }

        # Add paths for debugging (matching original implementation)
        sample["image_path"] = str(self.files[index]["image"])
        sample["label_path"] = str(self.files[index]["mask"])

        return sample

    def _load_image(self, index: int) -> tuple[Tensor, str]:
        """
        Load image with enhanced nodata handling for elevation data.

        Args:
            index: Index of the sample to load

        Returns:
            Tuple of (image_tensor, image_name)

        """
        image_path = self.files[index]["image"]
        image_name = Path(image_path).name

        with rio.open(image_path) as image:
            image_array = image.read().astype(np.float32)

            # Handle nodata values - set them to 0 (matching original implementation)
            if image.nodata is not None:
                image_array[image_array == image.nodata] = 0.0

            image_tensor = torch.from_numpy(image_array).float()

        return image_tensor, image_name

    def _load_mask(self, index: int) -> tuple[Tensor, str]:
        """
        Load mask with appropriate handling for segmentation labels.

        Args:
            index: Index of the sample to load

        Returns:
            Tuple of (mask_tensor, mask_name)

        """
        mask_path = self.files[index]["mask"]
        mask_name = Path(mask_path).name

        with rio.open(mask_path) as mask:
            mask_array = mask.read().astype(np.int64)

            # Handle nodata values - set them to -1 for ignore class
            if mask.nodata is not None:
                mask_array[mask_array == mask.nodata] = -1

            mask_tensor = torch.from_numpy(mask_array).float()

        return mask_tensor, mask_name
