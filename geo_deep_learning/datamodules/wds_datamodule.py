"""Multi-sensor data module for webdataset."""

import logging
from typing import Any

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from geo_deep_learning.datasets.wds_dataset import create_sharded_datasets
from geo_deep_learning.samplers.round_robin_sampler import create_round_robin_sampler

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
        sampling_strategy: str | dict[str, str] = "round_robin",
        sensor_weighting: str = "equal",
        custom_sensor_weights: dict[str, float] | None = None,
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
                - str: Same strategy for trn and val ("round_robin" or "mixed")
                - Dict: trn and val strategy {"trn":"round_robin", "val":"mixed"}
            sensor_weighting: How to weight sensors - "equal", "proportional", "custom"
            custom_sensor_weights: Custom weights if sensor_weighting="custom"
                - {"sensor_a": 4, "sensor_c": 1} gives sensor_a 4x more batches
            batch_size: Batch size for all dataloaders
            num_workers: Number of worker processes
            patch_size: Target patch size for augmentations

        """
        super().__init__()

        self.sensor_configs_path = sensor_configs_path
        self.model_type = model_type

        if isinstance(sampling_strategy, str):
            self.sampling_strategy = {
                "trn": sampling_strategy,
                "val": sampling_strategy,
            }
        else:
            self.sampling_strategy = sampling_strategy

        self.sensor_weighting = sensor_weighting
        self.custom_sensor_weights = custom_sensor_weights
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size

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
        self.datasets = create_sharded_datasets(
            sensor_configs_path=self.sensor_configs_path,
            model_type=self.model_type,
            transforms=self.transform,
            epoch_size=None,
        )

        self._create_combined_datasets()

    def _create_combined_datasets(self) -> None:
        """Create concatenated datasets for round-robin sampling."""
        self.combined_datasets = {}

        for split in ["trn", "val", "tst"]:
            sensor_datasets = {
                sensor_name: sensor_splits[split]
                for sensor_name, sensor_splits in self.datasets.items()
                if split in sensor_splits
            }
            if sensor_datasets:
                self.combined_datasets[split] = {
                    "dataset": ConcatDataset(list(sensor_datasets.values())),
                    "sensor_datasets": sensor_datasets,
                }

    def _create_round_robin_dataloader(self, split: str) -> DataLoader:
        """Create a round-robin DataLoader for the given split."""
        if split not in self.combined_datasets:
            msg = f"No combined dataset found for split: {split}"
            raise ValueError(msg)

        combined_data = self.combined_datasets[split]
        combined_dataset = combined_data["dataset"]
        sensor_datasets = combined_data["sensor_datasets"]

        # Determine if we're in distributed mode
        distributed = False
        if hasattr(self, "trainer") and self.trainer is not None:
            distributed = self.trainer.world_size > 1

        sampler = create_round_robin_sampler(
            datasets=sensor_datasets,
            batch_size=self.batch_size,
            sensor_weighting=self.sensor_weighting,
            custom_weights=self.custom_sensor_weights,
            distributed=distributed,
        )

        return DataLoader(
            combined_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_multisensor_batch,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def _create_mixed_dataloader(self, split: str) -> DataLoader:
        """Create a mixed DataLoader for the given split."""
        if split not in self.combined_datasets:
            msg = f"No combined dataset found for split: {split}"
            raise ValueError(msg)

        combined_dataset = self.combined_datasets[split]["dataset"]

        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=(split == "trn"),
            num_workers=self.num_workers,
            collate_fn=collate_multisensor_batch,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        strategy = self.sampling_strategy["trn"]

        if strategy == "round_robin":
            return self._create_round_robin_dataloader("trn")
        if strategy == "mixed":
            return self._create_mixed_dataloader("trn")

        msg = f"Unknown sampling strategy: {strategy}"
        raise ValueError(msg)

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        strategy = self.sampling_strategy["val"]

        if strategy == "round_robin":
            return self._create_round_robin_dataloader("val")
        if strategy == "mixed":
            return self._create_mixed_dataloader("val")

        msg = f"Unknown sampling strategy: {strategy}"
        raise ValueError(msg)

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if "tst" not in self.combined_datasets:
            return None
        return self._create_mixed_dataloader("tst")
