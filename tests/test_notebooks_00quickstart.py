"""Tests for the 00_quickstart notebook."""

from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Dataset

from geo_deep_learning.datamodules.csv_datamodule import CSVDataModule
from geo_deep_learning.tasks_with_models.segmentation_unetplus import (
    SegmentationUnetPlus,
)


# -------------------------
# Dummy Dataset / DataModule
# -------------------------
class RandomDataset(Dataset):
    """A dummy dataset yielding random images and masks."""

    def __len__(self) -> int:
        """Return the number of samples in the dataset (fixed to 4)."""
        return 4

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a random image and dummy mask for the given index."""
        x = torch.rand(3, 32, 32)  # RGB image
        y = torch.zeros(32, 32, dtype=torch.long)  # dummy mask
        return {"image": x, "mask": y}


class DummyDataModule(pl.LightningDataModule):
    """Lightweight DataModule returning random images and masks."""

    def train_dataloader(self) -> DataLoader:
        """Return DataLoader for training data."""
        return DataLoader(RandomDataset(), batch_size=2)

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for validation data."""
        return DataLoader(RandomDataset(), batch_size=2)

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for test data."""
        return DataLoader(RandomDataset(), batch_size=2)


# -------------------------
# Tests
# -------------------------
class TestSegmentationUnetPlus:
    """Tests for the SegmentationUnetPlus model."""

    def test_model_instantiates_as_lightningmodule(self) -> None:
        """Ensure SegmentationUnetPlus can be instantiated with minimal args."""
        model = SegmentationUnetPlus(
            encoder="resnet34",
            in_channels=3,
            num_classes=2,
            image_size=(64, 64),
            max_samples=1,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            scheduler=torch.optim.lr_scheduler.StepLR,
            scheduler_config={"step_size": 1, "gamma": 0.1},
            class_labels=["background", "buildings"],
            class_colors=["#000000", "#FF0000"],
        )
        assert hasattr(model, "training_step")
        assert isinstance(model, torch.nn.Module)


class TestCSVDataModule:
    """Tests for the CSVDataModule."""

    def test_datamodule_loads_csv_splits(self, tmp_path: Path) -> None:
        """Ensure CSVDataModule parses CSVs and sets up datasets."""
        # Create dummy CSVs for trn/val/tst
        for split in ["trn", "val", "tst"]:
            csv_path = tmp_path / f"{split}.csv"
            with csv_path.open("w") as f:
                f.write("image;mask\n")
                f.write("tests/data/dummy_img.tif;tests/data/dummy_lbl.tif\n")

        dm = CSVDataModule(
            batch_size=2,
            num_workers=0,
            mean=[0.5, 0.5, 0.5],
            std=[0.2, 0.2, 0.2],
            csv_root_folder=str(tmp_path),
            patches_root_folder=str(tmp_path),
        )
        dm.setup()
        assert dm.train_dataset is not None


class TestTrainerIntegration:
    """Integration tests for running Trainer with dummy data."""

    def test_trainer_runs_with_dummy_data(self) -> None:
        """Run 1-batch training loop with DummyDataModule (fast_dev_run)."""
        model = SegmentationUnetPlus(
            encoder="resnet34",
            in_channels=3,
            num_classes=2,
            image_size=(32, 32),
            max_samples=1,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1),
            scheduler_config={},
            class_labels=["background", "buildings"],
            class_colors=["#000000", "#FF0000"],
        )
        dm = DummyDataModule()
        trainer = Trainer(fast_dev_run=True, accelerator="cpu")
        trainer.fit(model, datamodule=dm)
