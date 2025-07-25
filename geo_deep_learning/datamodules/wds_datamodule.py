"""Multi-sensor data module for webdataset."""

import logging

import kornia as krn
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningDataModule
from webdataset import WebLoader

from geo_deep_learning.datasets.wds_dataset import create_sensor_datasets
from geo_deep_learning.datasets.wds_mixers import get_mixed_dataset

logger = logging.getLogger(__name__)


class MultiSensorDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for multi-sensor Earth observation data."""

    def __init__(  # noqa: PLR0913
        self,
        sensor_configs_path: str,
        model_type: str = "clay",
        patch_size: tuple[int, int] = (512, 512),
        epoch_size: int | None = None,
        batch_size: int = 16,
        num_workers: int = 8,
    ) -> None:
        """
        Initialize MultiSensorDataModule.

        Args:
            sensor_configs_path: Path to YAML config with sensor configurations
            model_type: Output format - "clay", "dofa", or "unified"
            patch_size: Target patch size for augmentations
            epoch_size: Number of patches per epoch
            batch_size: Batch size for all dataloaders
            num_workers: Number of worker processes

        """
        super().__init__()

        self.sensor_configs_path = sensor_configs_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.epoch_size = epoch_size
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
            batch_size=self.batch_size,
            epoch_size=self.epoch_size,
        )
        self.combined_datasets = {}
        for split in ["trn", "val", "tst"]:
            sensor_datasets = {
                sensor_name: sensor_splits[split]
                for sensor_name, sensor_splits in self.datasets.items()
                if split in sensor_splits
            }
            if sensor_datasets:
                self.combined_datasets[split] = get_mixed_dataset(
                    sensor_datasets=sensor_datasets,
                    strategy="round_robin",
                )

    def _create_dataloader(self, split: str) -> WebLoader:
        """Create a DataLoader for the given split."""
        if split not in self.combined_datasets:
            msg = f"No combined dataset found for split: {split}"
            raise ValueError(msg)
        return WebLoader(
            self.combined_datasets[split],
            num_workers=self.num_workers,
            batch_size=None,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self) -> WebLoader:
        """Create training DataLoader."""
        loader = self._create_dataloader("trn")
        return loader.with_epoch(self.epoch_size)

    def val_dataloader(self) -> WebLoader:
        """Create validation DataLoader."""
        return self._create_dataloader("val")

    def test_dataloader(self) -> WebLoader:
        """Create test DataLoader."""
        if "tst" not in self.combined_datasets:
            return None
        return self._create_dataloader("tst")

    def teardown(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Clean up after training/testing."""
        if hasattr(self, "datasets"):
            for sensor_datasets in self.datasets.values():
                for dataset in sensor_datasets.values():
                    if hasattr(dataset, "dataset") and dataset.dataset is not None:
                        dataset.dataset = None
