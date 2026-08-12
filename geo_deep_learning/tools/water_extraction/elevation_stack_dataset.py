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
_NON_STANDARDIZED_CHANNEL_KEYS = ("zero_intensity",)


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
        extra_input_rasters: list[str] | None = None,
        selected_channel_indices: list[int] | None = None,
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
        self.extra_input_rasters = extra_input_rasters or []
        self.selected_channel_indices = selected_channel_indices
        self.norm_stats = norm_stats or {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        }
        self._skip_standardization_indices = self._resolve_skip_standardization_indices()

        # Load files using custom method
        self.files = self._load_files()

        # Log dataset creation (using the same pattern as base class)
        log_dataset(self.split, len(self.files))

    def _resolved_channel_names(self) -> list[str]:
        """Return configured channel names in stack order."""
        channel_names = ["twi", "ndsm"]
        if self.include_intensity:
            channel_names.append("intensity")
        channel_names.extend(Path(raster).stem for raster in self.extra_input_rasters)

        if self.selected_channel_indices is None:
            return channel_names

        return [channel_names[idx] for idx in self.selected_channel_indices]

    def _resolve_skip_standardization_indices(self) -> list[int]:
        """Return channel indices that should bypass standardization."""
        skip_indices = []
        for idx, channel_name in enumerate(self._resolved_channel_names()):
            normalized_name = channel_name.lower()
            if any(key in normalized_name for key in _NON_STANDARDIZED_CHANNEL_KEYS):
                skip_indices.append(idx)
        return skip_indices

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

        standardize_indices = [
            idx for idx in range(num_channels)
            if idx not in self._skip_standardization_indices
        ]
        if standardize_indices:
            image[standardize_indices] = standardization(
                image[standardize_indices],
                mean[standardize_indices],
                std[standardize_indices],
            )

        # Guard: Check for non-finite values after preprocessing
        if not torch.isfinite(image).all():
            nan_count = torch.isnan(image).sum().item()
            inf_count = torch.isinf(image).sum().item()
            error_msg = (
                f"Non-finite values detected in preprocessed image {image_name}!\n"
                f"  NaN count: {nan_count}, Inf count: {inf_count}\n"
                f"  Raw image stats: min={image.min():.3f}, max={image.max():.3f}\n"
                f"  Normalization: mean={self.norm_stats['mean']}, std={self.norm_stats['std']}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

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

        Loads channels based on include_intensity and extra_input_rasters settings.

        Args:
            index: Index of the sample to load

        Returns:
            Tuple of (image_tensor, image_name)

        """
        image_path = self.files[index]["image"]
        image_name = Path(image_path).name

        # Determine which channels to load from the configured feature set.
        available_channels = 2 + int(self.include_intensity) + len(self.extra_input_rasters)
        if self.selected_channel_indices is None:
            channel_indices = list(range(available_channels))
        else:
            channel_indices = self.selected_channel_indices

        with rio.open(image_path) as image:
            total_bands = image.count

            if total_bands == len(channel_indices):
                channels_to_load = list(range(1, total_bands + 1))
            else:
                channels_to_load = [idx + 1 for idx in channel_indices]

            # Validate that the raster has enough bands
            if max(channels_to_load) > total_bands:
                error_msg = (f"Insufficient bands in {image_name}: "
                             f"requested channel indices {channel_indices} (include_intensity={self.include_intensity}, "
                             f"extra_input_rasters={self.extra_input_rasters}) "
                             f"but file only has {total_bands} bands")
                logger.error(error_msg)
                raise ValueError(error_msg)

            image_array = image.read(channels_to_load).astype(np.float32)

            # Handle nodata values - set them to 0.
            # Prefer formal nodata field; fall back to per-band NODATA_VALUE tags
            # (tiles written before rasterio nodata was set store the value there).
            if image.nodata is not None:
                image_array[image_array == image.nodata] = 0.0
            else:
                for i in range(image_array.shape[0]):
                    source_band_index = channels_to_load[i]
                    band_nodata_str = image.tags(source_band_index).get("NODATA_VALUE")
                    if band_nodata_str is not None:
                        try:
                            band_nodata = float(band_nodata_str)
                            image_array[i][image_array[i] == band_nodata] = 0.0
                        except (TypeError, ValueError):
                            pass

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
