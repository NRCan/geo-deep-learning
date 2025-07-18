"""Multi-sensor data module for webdataset."""

import logging
from typing import Any

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from geo_deep_learning.datasets.wds_dataset import create_sensor_datasets
from geo_deep_learning.datasets.wds_mixers import get_mixed_dataset

logger = logging.getLogger(__name__)


def collate_multisensor_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for multi-sensor batches.

    Args:
        batch: List of samples from one or more sensors

    Returns:
        Batched dictionary with stacked tensors

    """
    if not batch:
        return {}

    collated = {}

    sample_keys = batch[0].keys()

    for key in sample_keys:
        values = [sample[key] for sample in batch]

        if key in ["pixels", "mask", "time", "latlon", "wavelengths"]:
            try:
                collated[key] = torch.stack(values)
            except RuntimeError as e:
                logger.warning("Failed to stack %s: %s", key, e)
                collated[key] = values
        else:
            collated[key] = values

    return collated


class MultiSensorDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for multi-sensor Earth observation data."""

    def __init__(  # noqa: PLR0913
        self,
        sensor_configs_path: str,
        model_type: str = "clay",
        sampling_strategy: str | dict[str, str] = "round_robin",  # or "uniform"
        patch_size: tuple[int, int] = (512, 512),
        batch_size: int = 16,
        num_workers: int = 8,
    ) -> None:
        """
        Initialize MultiSensorDataModule.

        Args:
            sensor_configs_path: Path to YAML config with sensor configurations
            model_type: Output format - "clay", "dofa", or "unified"
            sampling_strategy: Sampling strategy per split. Can be:
                - str: Same strategy for trn and val ("round_robin" or "uniform")
                - Dict: trn and val strategy {"trn":"round_robin", "val":"uniform"}
            batch_size: Batch size for all dataloaders
            num_workers: Number of worker processes
            patch_size: Target patch size for augmentations

        """
        super().__init__()

        self.sensor_configs_path = sensor_configs_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size

        if isinstance(sampling_strategy, str):
            self.sampling_strategy = {
                "trn": sampling_strategy,
                "val": sampling_strategy,
            }
        else:
            self.sampling_strategy = sampling_strategy

        self.datasets = {}
        self.combined_datasets = {}

        random_resized_crop_zoom_in = krn.augmentation.RandomResizedCrop(
            size=self.patch_size,
            scale=(1.0, 2.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )
        random_resized_crop_zoom_out = krn.augmentation.RandomResizedCrop(
            size=self.patch_size,
            scale=(0.5, 1.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )

        self.transform = AugmentationSequential(
            krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomRotation90(
                times=(1, 3),
                p=0.5,
                align_corners=True,
                keepdim=True,
            ),
            random_resized_crop_zoom_in,
            random_resized_crop_zoom_out,
            data_keys=None,
            random_apply=1,
        )

    def prepare_data(self) -> None:
        """Prepare data."""

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Create datasets for each stage."""
        self.datasets = create_sensor_datasets(
            sensor_configs_path=self.sensor_configs_path,
            model_type=self.model_type,
            transforms=self.transform,
            epoch_size=10,
        )
        self.combined_datasets = {}
        for split in ["trn", "val", "tst"]:
            sensor_datasets = {
                sensor_name: sensor_splits[split]
                for sensor_name, sensor_splits in self.datasets.items()
                if split in sensor_splits
            }
            if sensor_datasets:
                strategy = self.sampling_strategy.get(split, "uniform")
                self.combined_datasets[split] = get_mixed_dataset(
                    sensor_datasets=sensor_datasets,
                    strategy=strategy,
                )

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create a round-robin DataLoader for the given split."""
        if split not in self.combined_datasets:
            msg = f"No combined dataset found for split: {split}"
            raise ValueError(msg)
        num_workers = self.num_workers

        if split == "trn":
            num_workers = min(2, self.num_workers)  # Max 2 for 2 shards
        elif split == "val":
            num_workers = min(1, self.num_workers)  # Max 1 for 1 shard
        else:  # test
            num_workers = min(4, self.num_workers)  # Max 4 for 4 shards

        return DataLoader(
            self.combined_datasets[split],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_multisensor_batch,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=(num_workers > 0),
            timeout=60,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return self._create_dataloader("trn")

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return self._create_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if "tst" not in self.combined_datasets:
            return None
        return self._create_dataloader("tst")
