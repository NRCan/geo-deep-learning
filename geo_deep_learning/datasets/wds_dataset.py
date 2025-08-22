"""Sharded WebDataset."""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import webdataset as wds
import yaml
from pytorch_lightning.utilities import rank_zero_only

from geo_deep_learning.tools.utils import normalization, standardization

logger = logging.getLogger(__name__)


@rank_zero_only
def log_dataset(
    sensor_name: str,
    split: str,
    shard_count: int | None = None,
    patch_count: int | None = None,
    *,
    valid: bool = False,
) -> None:
    """Log dataset(only on rank 0)."""
    if valid:
        logger.info(
            "Created dataset for %s %s split (%s shards) with %s patches",
            sensor_name,
            split,
            shard_count,
            patch_count,
        )
    else:
        logger.warning(
            "No shards found for %s %s split",
            sensor_name,
            split,
        )


def load_sensor_configs(config_path: str) -> dict[str, dict[str, str]]:
    """Load sensor configurations from a YAML file."""
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def create_shard_split_paths(
    manifest_path: str,
    split: str,
    parent_dir: str | None = None,
) -> tuple[list[str], int]:
    """
    Create shard split paths from a manifest file.

    Args:
        manifest_path: Path to the manifest file
        split: Split name (e.g., "trn", "val", "tst")
        parent_dir: Optional parent directory for the split shards

    Returns:
        Tuple of list of shard paths for the specified split and patch count

    """
    if parent_dir is None:
        shard_parent_path = Path(manifest_path).parent / split
    else:
        shard_parent_path = Path(parent_dir) / split
    with Path(manifest_path).open() as f:
        data = json.load(f)
    shard_data = data["shards"][split]
    patch_count = data["statistics"]["patch_counts"][split]
    return (
        [(shard_parent_path / item["path"]).as_posix() for item in shard_data],
        patch_count,
    )


def create_sensor_datasets(
    sensor_configs_path: str,
    **common_kwargs: object,
) -> dict[str, Any]:
    """
    Create multiple sensor datasets from configuration.

    Args:
        sensor_configs_path: Path to the sensor configurations YAML file
        **common_kwargs: Common arguments for all datasets

    Returns:
        Dict mapping sensor_name -> MultiSensorWebDataset

    """
    sensor_configs = load_sensor_configs(sensor_configs_path)
    datasets = {}
    for sensor_name, config in sensor_configs.items():
        try:
            datasets[sensor_name] = {}
            for split in ["trn", "val", "tst"]:
                shard_paths, patch_count = create_shard_split_paths(
                    manifest_path=config["manifest_path"],
                    split=split,
                    parent_dir=config["parent_dir"],
                )
                if len(shard_paths) == 0:
                    log_dataset(sensor_name, split, valid=False)
                    continue
                datasets[sensor_name][split] = ShardedDataset(
                    sensor_name=sensor_name,
                    shard_paths=shard_paths,
                    patch_count=patch_count,
                    normalization_stats_path=config["stats_path"],
                    split=split,
                    wavelength_keys=config.get("wavelength_keys"),
                    **common_kwargs,
                )

                log_dataset(
                    sensor_name,
                    split,
                    len(shard_paths),
                    patch_count,
                    valid=True,
                )
        except Exception:
            logger.exception(
                "Failed to create dataset for %s %s split",
                sensor_name,
                split,
            )

    return datasets


class ShardedDataset:
    """
    High-performance WebDataset for multi-sensor Earth observation data.

    - Sensor-specific normalization and standardization
    - Flexible foundation model formats (CLAY/DOFA/Unified)
    - Metadata parsing and encoding
    """

    def __init__(  # noqa: PLR0913
        self,
        sensor_name: str,
        shard_paths: list[str],
        patch_count: int,
        normalization_stats_path: str,
        model_type: str = "clay",  # "clay", "dofa", "unified"
        split: str = "trn",
        batch_size: int = 16,
        shuffle_buffer: int = 1000,
        shardshuffle: int | None = None,
        seed: int = 42,
        epoch_size: int | None = None,
        wavelength_keys: list[str] | None = None,
    ) -> None:
        """
        Initialize MultiSensorWebDataset.

        Args:
            sensor_name: Name of the sensor (e.g., "geoeye-1-ortho-pansharp_base")
            shard_paths: List of WebDataset shard URLs/paths
            patch_count: Number of patches in the dataset
            normalization_stats_path: Path to normalization statistics JSON
            model_type: Output format - "clay", "dofa", or "unified"
            split: Data split - "trn", "val", "tst"
            batch_size: Batch size
            shuffle_buffer: Number of batches to prefetch for shuffling
            shardshuffle: Number of shards to shuffle
            seed: Random seed for shuffling
            epoch_size: Size of epoch (for infinite streaming)
            wavelength_keys: Optional list of metadata keys for wavelengths

        """
        super().__init__()

        self.sensor_name = sensor_name
        self.shard_paths = shard_paths
        self.model_type = model_type
        self.split = split
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.shuffle_buffer = shuffle_buffer
        self.shardshuffle = shardshuffle
        self.patch_count = patch_count
        self.norm_stats = self._load_normalization_stats(normalization_stats_path)
        self.wavelength_keys = wavelength_keys
        self.wavelengths_cache = {}
        self.dataset = None
        self.seed = seed

    def _load_normalization_stats(self, stats_path: str) -> dict[str, Any]:
        """Load normalization statistics from JSON file."""
        with Path(stats_path).open() as f:
            data = json.load(f)

        stats = data["statistics"][self.sensor_name]
        mean = (
            torch.tensor(stats["mean"], dtype=torch.float32).div(255.0).view(-1, 1, 1)
        )
        std = torch.tensor(stats["std"], dtype=torch.float32).div(255.0).view(-1, 1, 1)

        return {
            "mean": mean,
            "std": std,
            "band_count": stats["band_count"],
            "patch_count": stats["patch_count"],
            "dtype": stats["dtype"],
        }

    def _process_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Process a single WebDataset sample.

        Expected sample structure:
        {
            "__key__": "sample_key",
            "image_patch.npy": numpy_array,
            "label_patch.npy": numpy_array,
            "metadata.json": metadata_dict
        }
        """
        # Extract data
        image = torch.from_numpy(sample["image_patch.npy"]).float()
        label = torch.from_numpy(sample["label_patch.npy"]).long()
        metadata = sample["metadata.json"]

        # Apply sensor-specific normalization
        image = normalization(image)
        image = standardization(image, self.norm_stats["mean"], self.norm_stats["std"])

        # Prepare output based on model type
        if self.model_type == "clay":
            return self._prepare_clay_output(image, label, metadata, sample["__key__"])
        if self.model_type == "dofa":
            return self._prepare_dofa_output(image, label, metadata, sample["__key__"])
        # unified
        return self._prepare_generic_output(image, label, metadata, sample["__key__"])

    def _prepare_clay_output(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        metadata: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        """Prepare output in CLAY format."""
        # Extract temporal information
        datetime_str = metadata["metadata"].get("datetime", "0.0")
        time_tensor = self._encode_temporal(datetime_str)
        # Extract spatial information
        lat = metadata["metadata"].get("coordinates_lat", 0.0)
        lon = metadata["metadata"].get("coordinates_lon", 0.0)
        latlon_tensor = self._encode_spatial(lat, lon)
        return {
            "image": image,
            "mask": label,
            "platform": self.sensor_name,
            "time": time_tensor,
            "latlon": latlon_tensor,
            "image_name": key,
            "mean": self.norm_stats["mean"],
            "std": self.norm_stats["std"],
        }

    def _prepare_dofa_output(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        metadata: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        """Prepare output in DOFA format."""
        wavelengths = self._extract_wavelengths(metadata)
        return {
            "image": image,
            "mask": label,
            "platform": self.sensor_name,
            "image_name": key,
            "wavelengths": wavelengths,
            "mean": self.norm_stats["mean"],
            "std": self.norm_stats["std"],
        }

    def _prepare_generic_output(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        metadata: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        """Prepare unified output with all metadata."""
        return {
            "image": image,
            "mask": label,
            "platform": self.sensor_name,
            "image_name": key,
            "metadata": metadata,
            "mean": self.norm_stats["mean"],
            "std": self.norm_stats["std"],
        }

    def _encode_temporal(self, datetime_str: str) -> torch.Tensor:
        """Encode temporal information using sine/cosine cyclical encoding."""
        try:
            # Handle UTC 'Z' suffix
            if datetime_str.endswith("Z"):
                datetime_str = datetime_str[:-1] + "+00:00"

            dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))

            # Week of year (1-52 or 53)
            week = dt.isocalendar().week
            hour = dt.hour

            # Convert to radians for cyclical encoding
            week_rad = (week / 52.0) * 2 * math.pi
            hour_rad = (hour / 24.0) * 2 * math.pi

            # Encode as sin/cos
            week_sin = math.sin(week_rad)
            week_cos = math.cos(week_rad)
            hour_sin = math.sin(hour_rad)
            hour_cos = math.cos(hour_rad)

            return torch.tensor(
                [week_sin, week_cos, hour_sin, hour_cos],
                dtype=torch.float32,
            )

        except Exception as e:  # noqa: BLE001
            logger.warning("Error parsing datetime: %s %s", datetime_str, e)
            return torch.zeros(4, dtype=torch.float32)

    def _encode_spatial(self, lat: float, lon: float) -> torch.Tensor:
        """Encode spatial (lat/lon) using sine/cosine cyclical encoding."""
        try:
            # Convert degrees to radians
            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)

            # Encode as sin/cos
            lat_sin = math.sin(lat_rad)
            lat_cos = math.cos(lat_rad)
            lon_sin = math.sin(lon_rad)
            lon_cos = math.cos(lon_rad)

            return torch.tensor(
                [lat_sin, lat_cos, lon_sin, lon_cos],
                dtype=torch.float32,
            )

        except Exception as e:  # noqa: BLE001
            logger.warning("Error parsing coordinates: %s %s %s", lat, lon, e)
            return torch.zeros(4, dtype=torch.float32)

    def _extract_wavelengths(self, metadata: dict[str, Any]) -> torch.Tensor:
        """Extract wavelength information for DOFA."""
        try:
            meta = metadata["metadata"]

            wavelengths_keys = self.wavelength_keys or [
                "red_wavelength",
                "green_wavelength",
                "blue_wavelength",
                "nir_wavelength",
            ]

            wavelengths = [
                float(meta[band]) for band in wavelengths_keys if band in meta
            ]

            cache_key = f"{self.sensor_name}_{'_'.join(wavelengths_keys)}"

            if cache_key not in self.wavelengths_cache:
                self.wavelengths_cache[cache_key] = torch.tensor(
                    wavelengths,
                    dtype=torch.float32,
                )

            return self.wavelengths_cache[cache_key]

        except Exception as e:  # noqa: BLE001
            logger.warning("Error extracting wavelengths: %s", e)
            return torch.tensor([0.0] * len(wavelengths_keys), dtype=torch.float32)

    def build_web_dataset(self) -> wds.WebDataset:
        """Create optimized WebDataset pipeline for HPC."""
        shard_list = sorted(self.shard_paths)

        if self.split == "trn":
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                shard_list = shard_list[rank::world_size]
            dataset = wds.WebDataset(
                urls=shard_list,
                shardshuffle=self.shardshuffle,
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
                empty_check=False,
                seed=self.seed,
            )
            dataset = dataset.shuffle(self.shuffle_buffer)
        else:
            dataset = wds.WebDataset(
                urls=shard_list,
                shardshuffle=False,
                nodesplitter=None if self.split == "tst" else wds.split_by_node,
                workersplitter=wds.split_by_worker,
                empty_check=False,
            )
        return (
            dataset.decode()
            .map(self._process_sample, handler=wds.warn_and_continue)
            .batched(self.batch_size, partial=self.split != "trn")
        )
