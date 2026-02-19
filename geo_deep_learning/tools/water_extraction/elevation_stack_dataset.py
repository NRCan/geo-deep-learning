"""Elevation Stack dataset that extends CSVDataset for water extraction tasks."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from torch import Tensor

from geo_deep_learning.datasets.csv_dataset import CSVDataset, log_dataset
from geo_deep_learning.utils.tensors import standardization

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
        csv_path: str | None = None,
        csv_infer_path: str | None = None,
        include_intensity: bool = True,
    ) -> None:
        """
        Initialize the ElevationStackDataset.

        Args:
            csv_root_folder (str): The root folder where the csv file is stored
            patches_root_folder (str): The root folder of image and mask patches.
            split (str, optional): Data split ("trn", "val", "tst"). Defaults to "trn".
            norm_stats (dict[str, list[float]], optional): Normalization statistics.
                Should contain "mean" and "std" keys with lists of per-band values.
            include_intensity (bool, optional): Whether to load intensity band. Defaults to True.

        """
        # Initialize parent class attributes first
        self.csv_root_folder = csv_root_folder
        self.patches_root_folder = patches_root_folder
        self.split = split
        self.csv_path = csv_path
        self.csv_infer_path = csv_infer_path
        self.include_intensity = include_intensity
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

        Overrides the base class method to handle CSV with "tif", "gpkg",
        and "split" columns.

        Returns:
            List of dictionaries with "image" and "mask" keys.

        """
        # Select the correct CSV path
        if self.split == "inference":
            logger.info("INFERENCE CSV: %s", self.csv_infer_path)
            csv_path = Path(self.csv_infer_path)
        else:
            csv_path = Path(self.csv_path)

        if not csv_path.exists():
            msg = f"CSV file not found: {csv_path}"
            raise FileNotFoundError(msg)

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
        # row = self.df.iloc[index] # DEBUG

        image, image_name = self._load_image(index)
        mask, mask_name = self._load_mask(index)

        # Apply normalization (0-1 scaling)
        # image = normalization(image)

        # Validate channel count matches stats
        num_channels = image.shape[0]
        num_stats = len(self.norm_stats["mean"])
        
        if num_channels != num_stats:
            error_msg = (
                f"Channel mismatch in {image_name}: "
                f"image has {num_channels} channels but "
                f"normalization stats have {num_stats} values. "
                f"Stats: mean={self.norm_stats['mean']}, std={self.norm_stats['std']}. "
                f"include_intensity={self.include_intensity}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Apply standardization using provided statistics
        mean = torch.tensor(self.norm_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.norm_stats["std"], dtype=torch.float32).view(-1, 1, 1)
        
        # Debug logging for first few samples
        # if index < 3:
        #     logger.info(
        #         f"Sample {index} ({image_name}): "
        #         f"channels={num_channels}, "
        #         f"include_intensity={self.include_intensity}, "
        #         f"mean={self.norm_stats['mean']}, "
        #         f"std={self.norm_stats['std']}, "
        #         f"split={self.split}"
        #     )
        
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
        
        Loads channels based on include_intensity setting:
        - include_intensity=True: loads all 3 channels [TWI, nDSM, Intensity]
        - include_intensity=False: loads only 2 channels [TWI, nDSM]

        Args:
            index: Index of the sample to load

        Returns:
            Tuple of (image_tensor, image_name)

        """
        image_path = self.files[index]["image"]
        image_name = Path(image_path).name

        # Determine how many channels to load based on include_intensity
        num_channels = 3 if self.include_intensity else 2

        with rio.open(image_path) as image:
            total_bands = image.count
            
            # Validate that the raster has enough bands
            if total_bands < num_channels:
                error_msg = (f"Insufficient bands in {image_name}: "
                             f"expected {num_channels} bands (include_intensity={self.include_intensity}) "
                             f"but file only has {total_bands} bands")
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Load only the required number of channels (1-indexed in rasterio)
            if num_channels < total_bands:
                # Load subset of bands: [1, 2] for 2 channels, [1, 2, 3] for 3 channels
                channels_to_load = list(range(1, num_channels + 1))
                image_array = image.read(channels_to_load).astype(np.float32)
                # if index < 3:  # Debug logging for first few samples
                #     logger.info("Loading %d/%d channels from %s (include_intensity=%s, split=%s)",
                #         num_channels,
                #         total_bands,
                #         image_name,
                #         self.include_intensity,
                #         self.split,
                #     )
            else:
                # Load all bands
                image_array = image.read().astype(np.float32)
                if index < 3:
                    logger.info(
                        "Loading all %d channels from %s (include_intensity=%s, split=%s)",
                        total_bands,
                        image_name,
                        self.include_intensity,
                        self.split,
                    )

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
