"""Multi-sensor data module for webdataset."""

import logging

import webdataset as wds
from lightning.pytorch import LightningDataModule
from webdataset import WebLoader

from geo_deep_learning.datasets.wds_dataset import create_sensor_datasets

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
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def prepare_data(self) -> None:
        """Prepare data."""

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Create datasets for each stage."""
        self.datasets = create_sensor_datasets(
            sensor_configs_path=self.sensor_configs_path,
            model_type=self.model_type,
            batch_size=self.batch_size,
            epoch_size=self.epoch_size,
        )
        self._setup_train_loader()
        self._setup_val_loader()
        self._setup_test_loader()

    def _setup_train_loader(self) -> None:
        """Set up training loader with sensor mixing."""
        train_datasets = {}
        for sensor_name, splits in self.datasets.items():
            if "trn" in splits:
                train_datasets[sensor_name] = splits["trn"]

        if not train_datasets:
            logger.warning("No training datasets found!")
            return

        if len(train_datasets) == 1:
            # Single sensor
            sensor_name = next(iter(train_datasets.keys()))
            sensor_dataset = train_datasets[sensor_name].build_web_dataset()
        else:
            # Multiple sensors
            train_datasets = {
                name: dataset.build_web_dataset()
                for name, dataset in train_datasets.items()
            }
            sensor_dataset = self._create_mixed_dataset(train_datasets)

        self.train_loader = WebLoader(
            sensor_dataset,
            num_workers=self.num_workers,
            batch_size=None,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=(self.num_workers > 0),
        )
        if self.epoch_size:
            self.train_loader = self.train_loader.with_epoch(self.epoch_size)

    def _setup_val_loader(self) -> None:
        """Set up validation loader."""
        val_datasets = {}
        for sensor_name, splits in self.datasets.items():
            if "val" in splits:
                val_datasets[sensor_name] = splits["val"]

        if not val_datasets:
            logger.warning("No validation datasets found!")
            return

        if len(val_datasets) == 1:
            # Single sensor
            sensor_name = next(iter(val_datasets.keys()))
            sensor_dataset = val_datasets[sensor_name].build_web_dataset()
        else:
            # Multiple sensors
            val_datasets = {
                name: dataset.build_web_dataset()
                for name, dataset in val_datasets.items()
            }
            sensor_dataset = self._create_mixed_dataset(val_datasets)

        self.val_loader = WebLoader(
            sensor_dataset,
            num_workers=self.num_workers,
            batch_size=None,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=(self.num_workers > 0),
        )

    def _setup_test_loader(self) -> None:
        """Set up test loader."""
        test_datasets = {}
        for sensor_name, splits in self.datasets.items():
            if "tst" in splits:
                test_datasets[sensor_name] = splits["tst"]

        if not test_datasets:
            logger.info("No test datasets found - this is optional")
            return

        if len(test_datasets) == 1:
            # Single sensor
            sensor_name = next(iter(test_datasets.keys()))
            sensor_dataset = test_datasets[sensor_name].build_web_dataset()
        else:
            # Multiple sensors
            test_datasets = {
                name: dataset.build_web_dataset()
                for name, dataset in test_datasets.items()
            }
            sensor_dataset = self._create_mixed_dataset(test_datasets)

        self.test_loader = WebLoader(
            sensor_dataset,
            num_workers=self.num_workers,
            batch_size=None,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=(self.num_workers > 0),
        )

    def _create_mixed_dataset(
        self,
        sensor_datasets: dict[str, wds.WebDataset],
    ) -> wds.WebDataset:
        """Create a mixed dataset from multiple sensor datasets."""
        datasets_list = list(sensor_datasets.values())
        # Round robin through sensors
        return wds.RandomMix(
            datasets=datasets_list,
            probs=None,  # Equal probability
            longest=True,
        )

    def train_dataloader(self) -> WebLoader:
        """Create training DataLoader."""
        return self.train_loader

    def val_dataloader(self) -> WebLoader:
        """Create validation DataLoader."""
        return self.val_loader

    def test_dataloader(self) -> WebLoader:
        """Create test DataLoader."""
        return self.test_loader

    def teardown(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Clean up after training/testing."""
        if hasattr(self, "datasets"):
            self.datasets.clear()
